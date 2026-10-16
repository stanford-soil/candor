#!/usr/bin/env python
"""
Token-level conditional surprisals for every turn of every conversation, using minicons.

For each turn t in a conversation, every token of the utterance is scored given
  (a) all previous turns of the conversation (as far as the context window allows), and
  (b) the formatting tokens that precede it (chat-template role headers, BOS, ...).

Output: one row per *content* token (template/special tokens are used as context but
are never written out), in long format, so that aggregation (turn mean/sum, word-level,
first-N-tokens, ...) can be done in analysis. Join back to the original CSV on
(transcript_id, turn_id).

Output columns
--------------
transcript_id, turn_id, speaker      identifiers from the input
role                                 role used in the prompt ("user"/"assistant" in chat mode,
                                     "A"/"B" in plain mode). By default the FIRST speaker of each
                                     conversation is "user"; --swap-roles makes the first speaker
                                     "assistant" (an empty user turn is then inserted in chat mode).
window_id                            index of the forward pass that produced this turn's scores
n_prior_tokens                       number of tokens (incl. template) before this turn in that window;
                                     0-context sensitivity check for long conversations
token_idx                            0-based position within the turn
token_id, token                      token id and its string (tokenizer vocabulary form)
char_start, char_end                 character span of the token within the (whitespace-normalised)
                                     utterance; use for mapping tokens -> words
surprisal                            -log p(token | context), in bits (or nats with --nats)
model, context, format_mode, unit,   run settings, constant within a file (also in <output>.meta.json
bow_correction, swap_roles, skip_turn_zero,          together with dtype, max_tokens, system prompt, minicons version)

Usage
-----
python conversation_surprisal.py \
    --input dense_subset1_and_subset2_processed.csv \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output surprisals_llama31_8b.csv.gz \
    --device cuda:0 --dtype bfloat16 --hf-home /scratch/me/hf_home

Base (non-chat) models are formatted as plain dialogue ("A: ...\\nB: ...\\n").
Chat/instruct models use tokenizer.apply_chat_template. --mode auto picks chat iff the
tokenizer ships a chat template. The run is resumable: re-running with the same --output
skips transcripts that are already in the file.

Options
-------
--skip-turn-zero   drop turn 0 of every conversation (not scored, not used as context)
--swap-roles       first speaker = assistant, second = user
--context both     global AND local scores in one run (global -> --output, local -> --local-output or
                   <output>_local.csv.gz); the analysis notebook computes PMI = local - global from them
"""
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

# ---- SET ME: directory where Hugging Face model weights / caches are stored ----------------
# Can also be given on the command line with --hf-home. If left as the placeholder below and
# no --hf-home is passed, the existing HF_HOME environment variable (or HF default) is used.
load_dotenv()
HF_HOME = os.getenv("HF_HOME")
HF_TOKEN = os.getenv("HF_TOKEN")
# -------------------------------------------------------------------------------------------

PLACEHOLDER = "[[[TURN_{:06d}]]]"
PLACEHOLDER_RE = re.compile(r"\[\[\[TURN_(\d{6})\]\]\]")
ROLE_NAMES = ("user", "assistant")
PLAIN_LABELS = ("A", "B")


# --------------------------------------------------------------------------- data
def load_conversations(path, id_col, turn_col, speaker_col, text_col):
    df = pd.read_csv(path, usecols=[id_col, turn_col, speaker_col, text_col], dtype={id_col: str})
    df = df.rename(columns={id_col: "transcript_id", turn_col: "turn_id",
                            speaker_col: "speaker", text_col: "utterance"})
    df["utterance"] = (df["utterance"].fillna("").astype(str)
                       .map(lambda s: re.sub(r"\s+", " ", s).strip()))

    # The file can contain several rows per (transcript_id, turn_id) (e.g. one per backchannel).
    # The utterance must be identical across them; keep one.
    key = ["transcript_id", "turn_id"]
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        n_diff = (dup.groupby(key)["utterance"].nunique() > 1).sum()
        if n_diff:
            warnings.warn(f"{n_diff} duplicated (transcript_id, turn_id) keys have differing "
                          "utterances; keeping the first occurrence.")
        print(f"[data] dropping {df.duplicated(key).sum()} duplicate turn rows", file=sys.stderr)
    df = df.drop_duplicates(key, keep="first").sort_values(key, kind="stable")
    return df.reset_index(drop=True)


def assign_roles(speakers, transcript_id, swap=False):
    """First speaker -> role 0 ("user"), second -> role 1 ("assistant"); swap=True reverses this.
    Requires strict alternation."""
    order = list(dict.fromkeys(speakers))
    if len(order) != 2:
        raise ValueError(f"[{transcript_id}] expected exactly 2 speakers, found {len(order)}")
    role_of = {order[0]: int(swap), order[1]: int(not swap)}
    roles = [role_of[s] for s in speakers]
    if any(a == b for a, b in zip(roles, roles[1:])):
        raise ValueError(f"[{transcript_id}] consecutive turns by the same speaker; chat templates "
                         "need strict alternation. Merge those turns upstream.")
    return roles


# ------------------------------------------------------------------- formatting
class Formatter:
    """Turns a list of (role, utterance) into model input text + character spans of each utterance.

    Chat mode: the conversation is rendered with the tokenizer's own chat template using
    placeholder contents; the rendered string is then cut at the placeholders, so we get the
    template's exact "skeleton" (headers, separators, end-of-turn tokens) and can slot the real
    utterances in with known character offsets. No guessing/regex over the utterance text.
    """

    def __init__(self, tokenizer, mode="auto", system_prompt=None):
        self.tok = tokenizer
        if mode == "auto":
            mode = "chat" if getattr(tokenizer, "chat_template", None) else "plain"
        if mode == "chat" and not getattr(tokenizer, "chat_template", None):
            raise ValueError("--mode chat requested but the tokenizer has no chat template")
        self.mode = mode
        self.system_prompt = system_prompt
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required (character offsets are used).")
        if mode == "plain":
            self.prefix_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None \
                else tokenizer.eos_token_id
            if self.prefix_id is None:
                raise ValueError("Tokenizer has neither BOS nor EOS token for plain mode")

    # -- skeleton
    def _skeleton(self, roles):
        n = len(roles)
        if self.mode == "chat":
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs += [{"role": ROLE_NAMES[r], "content": PLACEHOLDER.format(i)}
                     for i, r in enumerate(roles)]
            rendered = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            rendered = "".join(f"{PLAIN_LABELS[r]}: {PLACEHOLDER.format(i)}\n"
                               for i, r in enumerate(roles))
        parts = PLACEHOLDER_RE.split(rendered)  # [pre, idx0, mid, idx1, ..., post]
        idxs = [int(x) for x in parts[1::2]]
        if idxs != list(range(n)):
            raise RuntimeError("Chat template dropped, duplicated or reordered message contents; "
                               "cannot align turns. Template output was:\n" + rendered[:500])
        return parts[0::2]  # n + 1 template pieces

    # -- encode
    def encode(self, roles, utterances):
        """Returns ids, offsets, turn_token_ids where turn_token_ids[i] = positions of turn i's content tokens."""
        # Chat templates (Llama-2, Mistral, Gemma, ...) require the first message to be from the user.
        # If the first turn belongs to the assistant role (--swap-roles, or a window starting on role 1),
        # an EMPTY user turn is prepended; it is context only and is never returned.
        pad = self.mode == "chat" and roles[0] == 1
        if pad:
            roles, utterances = [0] + list(roles), [""] + list(utterances)
        pieces = self._skeleton(roles)
        text, spans = "", []
        for i, u in enumerate(utterances):
            text += pieces[i]
            spans.append((len(text), len(text) + len(u)))
            text += u
        text += pieces[-1]

        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        if self.mode == "plain":
            ids, offs = [self.prefix_id] + ids, [(0, 0)] + offs

        turn_pos = [[] for _ in utterances]
        t = 0
        for k, (s, e) in enumerate(offs):
            if e <= s:  # zero-width (BOS etc.)
                continue
            while t < len(spans) and spans[t][1] <= s:
                t += 1
            if t == len(spans):
                break
            cs, ce = spans[t]
            if s < ce and e > cs:  # token overlaps the utterance span
                turn_pos[t].append(k)
        for t, (u, pos) in enumerate(zip(utterances, turn_pos)):
            if u and not pos:
                raise RuntimeError(f"No tokens aligned to turn {t} ({u[:40]!r})")
            if pos and pos[0] == 0:
                raise RuntimeError("First sequence position is a content token; it would receive no score.")
        if pad:
            spans, turn_pos = spans[1:], turn_pos[1:]
        return ids, offs, spans, turn_pos

    def count_overhead(self):
        """Approximate template tokens added per turn (used only for window planning)."""
        probe = ["x"] * 6
        roles = [0, 1] * 3
        ids, *_ = self.encode(roles, probe)
        x = len(self.tok("x", add_special_tokens=False)["input_ids"])
        return max(1, -(-(len(ids) - 6 * x) // 6)) + 2  # ceil + safety margin


# ------------------------------------------------------------------- windowing
def window_bounds(costs, scored, max_tokens, context_tokens):
    """Next turn-aligned window given that turns [0, scored) are already scored.

    Returns (start, end): turns [start, scored) are context only, turns [scored, end) get scored.
    (Windows may start on either role; Formatter.encode pads an empty user turn when needed.)
    If the whole conversation fits, there is exactly one window and every turn sees its full history.
    """
    n = len(costs)
    start, ctx = scored, 0
    while start > 0 and ctx + costs[start - 1] <= context_tokens:
        start -= 1
        ctx += costs[start]
    end, total = scored, ctx
    while end < n and total + costs[end] <= max_tokens:
        total += costs[end]
        end += 1
    if end == scored:
        raise ValueError(f"Turn {scored} alone ({costs[scored]} tokens) exceeds the token limit {max_tokens}")
    return start, end


# ------------------------------------------------------------------- scoring
def score_conversation(scorer, fmt, tid, speakers, utterances, max_tokens, context_tokens,
                       base_two=True, bow_correction=False, context="global", swap_roles=False):
    """context="global": each turn conditioned on the preceding conversation (windowed if too long).
    context="local":  each turn scored ALONE (only its own role header + preceding tokens of the same
                      turn). Assistant-role turns get an empty user turn in front (see Formatter.encode) so
                      the template stays valid and the role header is identical to the global run.
                      Used for PMI: PMI = local surprisal - global surprisal."""
    import torch
    from transformers import BatchEncoding

    def run_model(ids):
        batch = BatchEncoding({"input_ids": torch.tensor([ids]),
                               "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        res = scorer.token_score(batch, surprisal=True, base_two=base_two,
                                 bow_correction=bow_correction)[0]  # [(token, surprisal)]; pos 0 is padded 0.0
        assert len(res) == len(ids)
        return res

    def emit(rows, local, gidx, w_id, ids, offs, spans, turn_pos, res):
        u = utterances[gidx]
        cs = spans[local][0]
        first_pos = turn_pos[local][0] if turn_pos[local] else len(ids)
        for j, k in enumerate(turn_pos[local]):
            s, e = offs[k]
            rows.append({
                "transcript_id": tid,
                "turn_id": None,  # filled by caller
                "speaker": speakers[gidx],
                "role": ROLE_NAMES[roles[gidx]] if fmt.mode == "chat" else PLAIN_LABELS[roles[gidx]],
                "_turn_index": gidx,
                "window_id": w_id,
                "n_prior_tokens": first_pos,
                "token_idx": j,
                "token_id": ids[k],
                "token": res[k][0],
                "char_start": max(s - cs, 0),
                "char_end": min(e - cs, len(u)),
                "surprisal": res[k][1],
            })

    roles = assign_roles(speakers, tid, swap_roles)
    if context == "local":
        rows = []
        for i, u in enumerate(utterances):
            ids, offs, spans, turn_pos = fmt.encode([roles[i]], [u])
            if len(ids) > max_tokens:
                raise ValueError(f"[{tid}] turn {i} does not fit in the token limit {max_tokens}")
            emit(rows, 0, i, -1, ids, offs, spans, turn_pos, run_model(ids))
        return rows

    overhead = fmt.count_overhead()
    costs = [len(fmt.tok(u, add_special_tokens=False)["input_ids"]) + overhead for u in utterances]
    # fixed prefix (BOS / system prompt) is charged once per window
    fixed = len(fmt.encode([0], [""])[0]) + 8 + overhead  # + room for a padded empty user turn
    rows, first, w_id = [], 0, 0
    while first < len(utterances):
        start, end = window_bounds(costs, first, max_tokens - fixed, context_tokens)
        while True:  # shrink if the true length (vs. the estimate) exceeds the limit
            ids, offs, spans, turn_pos = fmt.encode(roles[start:end], utterances[start:end])
            if len(ids) <= max_tokens or end - first <= 1:
                break
            end -= 1
        if len(ids) > max_tokens:
            raise ValueError(f"[{tid}] turn {first} does not fit in the token limit {max_tokens}")

        res = run_model(ids)
        for local in range(first - start, end - start):
            emit(rows, local, start + local, w_id, ids, offs, spans, turn_pos, res)
        first, w_id = end, w_id + 1
    return rows


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="CSV path (.csv or .csv.gz)")
    ap.add_argument("--model", required=True, help="HF model name or local path (causal LM)")
    ap.add_argument("--hf-home", default=HF_HOME,
                    help="directory for HF_HOME (model weights/cache); overrides the HF_HOME constant "
                         "at the top of this script")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--mode", default="auto", choices=["auto", "chat", "plain"])
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max sequence length per forward pass (clipped to the model's limit)")
    ap.add_argument("--context-tokens", type=int, default=1024,
                    help="for conversations longer than --max-tokens: min. preceding context kept per window")
    ap.add_argument("--context", default="global", choices=["global", "local", "both"],
                    help="global: condition on the whole preceding conversation (default). "
                         "local: score each turn on its own (no prior turns). "
                         "both: do both in one run; global goes to --output, local to --local-output "
                         "(PMI = local - global is computed by analyze_surprisal.ipynb from the two files)")
    ap.add_argument("--local-output", default=None,
                    help="output path for the local scores when --context both "
                         "(default: --output with '_local' inserted before the extension)")
    ap.add_argument("--skip-turn-zero", action="store_true",
                    help="drop turn_id 0 of every conversation (often a pooled block of backchannels). "
                         "It is neither scored nor used as context; the first remaining speaker becomes the first role")
    ap.add_argument("--swap-roles", action="store_true",
                    help="first speaker = assistant, second = user (default: first = user). In chat mode an "
                         "empty user turn is inserted in front where the template needs a user turn first")
    ap.add_argument("--nats", action="store_true", help="report nats instead of bits")
    ap.add_argument("--bow-correction", action="store_true",
                    help="minicons' beginning-of-word correction (Pimentel & Meister 2024; Oh & Schuler 2024)")
    ap.add_argument("--limit-transcripts", type=int, default=None)
    ap.add_argument("--id-col", default="transcript_id")
    ap.add_argument("--turn-col", default="turn_id")
    ap.add_argument("--speaker-col", default="speaker")
    ap.add_argument("--text-col", default="utterance")
    args = ap.parse_args()

    # HF_HOME must be set BEFORE transformers / huggingface_hub are imported (they read it at import time)
    if args.hf_home and args.hf_home != HF_HOME_PLACEHOLDER:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser())
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    print(f"[hf] HF_HOME={os.environ.get('HF_HOME', '(default: ~/.cache/huggingface)')}", file=sys.stderr)

    from minicons import scorer as mc_scorer

    kwargs = {}
    if args.dtype:
        import torch
        kwargs["torch_dtype"] = getattr(torch, args.dtype)
    lm = mc_scorer.IncrementalLMScorer(args.model, device=args.device, **kwargs)

    limit = args.max_tokens
    model_limit = getattr(lm.model.config, "max_position_embeddings", None)
    if model_limit and model_limit < limit:
        limit = model_limit
    fmt = Formatter(lm.tokenizer, args.mode, args.system_prompt)
    print(f"[model] {args.model} | mode={fmt.mode} | context={args.context} | max_tokens={limit}", file=sys.stderr)

    df = load_conversations(args.input, args.id_col, args.turn_col, args.speaker_col, args.text_col)
    if args.skip_turn_zero:
        n0 = int((df["turn_id"] == 0).sum())
        df = df[df["turn_id"] != 0].reset_index(drop=True)
        print(f"[data] --skip-turn-zero: dropped {n0} turn-0 rows", file=sys.stderr)
    tids = list(dict.fromkeys(df["transcript_id"]))
    if args.limit_transcripts:
        tids = tids[: args.limit_transcripts]

    # Settings stamped onto every output row (and written to <output>.meta.json)
    try:
        import minicons
        minicons_version = getattr(minicons, "__version__", None)
    except Exception:
        minicons_version = None
    contexts = ["global", "local"] if args.context == "both" else [args.context]
    out_paths = {}
    for c in contexts:
        if args.context == "both" and c == "local":
            if args.local_output:
                out_paths[c] = Path(args.local_output)
            else:
                o = Path(args.output)
                stem = o.name
                for ext in (".csv.gz", ".csv"):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)] + "_local" + ext
                        break
                else:
                    stem += "_local"
                out_paths[c] = o.with_name(stem)
        else:
            out_paths[c] = Path(args.output)

    base_meta = {
        "model": args.model,
        "format_mode": fmt.mode,            # chat | plain
        "unit": "nats" if args.nats else "bits",
        "bow_correction": bool(args.bow_correction),
        "swap_roles": bool(args.swap_roles),
        "skip_turn_zero": bool(args.skip_turn_zero),
    }
    done, headers, metas = {}, {}, {}
    for c, out in out_paths.items():
        run_meta = {**base_meta, "context": c}   # global | local
        metas[c] = run_meta
        extra_meta = {**run_meta, "dtype": args.dtype, "max_tokens": limit, "context_tokens": args.context_tokens,
                      "system_prompt": args.system_prompt,
                      "has_chat_template": bool(getattr(lm.tokenizer, "chat_template", None)),
                      "minicons_version": minicons_version, "input": str(args.input)}
        done[c] = set()
        if out.exists():
            prev = pd.read_csv(out, nrows=1, dtype=str)
            for k, v in run_meta.items():
                if k in prev.columns and str(prev[k].iloc[0]) != str(v):
                    raise SystemExit(f"[resume] {out} was produced with {k}={prev[k].iloc[0]!r}, "
                                     f"but this run has {k}={v!r}. Use a different --output.")
            done[c] = set(pd.read_csv(out, usecols=["transcript_id"], dtype=str)["transcript_id"])
            print(f"[resume] {len(done[c])} transcripts already in {out}", file=sys.stderr)
        headers[c] = not out.exists()
        Path(str(out) + ".meta.json").write_text(json.dumps(extra_meta, indent=2))

    for n_done, tid in enumerate(tids, 1):
        d = df[df["transcript_id"] == tid]
        turn_ids = d["turn_id"].tolist()
        for c in contexts:
            if tid in done[c]:
                continue
            rows = score_conversation(
                lm, fmt, tid, d["speaker"].tolist(), d["utterance"].tolist(),
                limit, min(args.context_tokens, limit // 2),
                base_two=not args.nats, bow_correction=args.bow_correction, context=c,
                swap_roles=args.swap_roles)
            if not rows:
                continue
            res = pd.DataFrame(rows)
            res["turn_id"] = res.pop("_turn_index").map(lambda i: turn_ids[i])
            for k, v in metas[c].items():
                res[k] = v
            res.to_csv(out_paths[c], mode="a", header=headers[c], index=False)  # gzip appends as new member; pandas reads it fine
            headers[c] = False
            print(f"[{n_done}/{len(tids)}] {tid} ({c}): {len(d)} turns, {len(res)} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
