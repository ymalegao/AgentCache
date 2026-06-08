#!/usr/bin/env python3
"""
LLM-as-a-judge code quality evaluation for the multi-turn benchmark.

Pairwise comparison of synthetic (centroid) vs cold (full system prompt)
responses, judged by an OpenAI model. See llm_judge_design.md for the full
rationale, system prompt, and interpretation guide.

Scope (per the data hygiene findings in llm_judge_design.md Sec 3.3):
  - Qwen-7B  (qwen7b.jsonl)                 : ALL 10 turns x 3 N = 30 pairs.
                                              Clean in both text and generation
                                              context (no Harmony channels).
  - GPT-OSS-20B (gptmulti_turn_benchmark)   : TURN 1 ONLY x 3 N = 3 pairs.
                                              The history-feedback bug corrupted
                                              the generation context of turns >1,
                                              so only the no-history cold-start
                                              turn is defensible. Cold turn-1 text
                                              is recovered via final-channel strip.

Usage:
    # build pairs and print counts, no API calls
    python judge_multi_turn.py --dry-run

    # run the judge (needs OPENAI_API_KEY in the environment)
    python judge_multi_turn.py --judge-model gpt-5.5

    # judge only one source
    python judge_multi_turn.py --only qwen7b
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_RESULTS = _EXP / "results"


# ---------------------------------------------------------------------------
# Judge prompt (mirrors llm_judge_design.md Sec 5 — publish this in the appendix)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a senior Python engineer serving as an impartial judge. You will be shown
two transcripts of the same multi-turn coding conversation, labeled A and B. Both
transcripts contain the same user requests, but the assistant responses differ.
Your job is to compare ONLY the final assistant response in each transcript - the
answer to the last user request - and decide which one better serves the user,
or whether they are equivalent in quality.

Use the earlier turns of each transcript purely as context for understanding what
the final request refers to (e.g. "add a flag to the script" refers to code built
up in that transcript's earlier turns). Judge each final response against its OWN
transcript's history. Do not penalize a final response because the two transcripts
diverged in earlier turns, and do not judge the quality of earlier responses -
they are context, not the object of evaluation.

Evaluate the two final responses on these four dimensions, in this order of
importance:

1. correctness - Would the code run as written and do what the user asked?
   Look for bugs, wrong APIs, logic errors, and code that would raise exceptions
   on the described inputs. For a modification request, correctness includes
   correctly integrating with the code established earlier in that transcript.
2. completeness - Does the response address every part of the final request? A
   request may contain multiple requirements (e.g. read a file AND print a
   summary AND handle a specific column type).
3. code_quality - Is the code idiomatic, readable, and reasonably robust
   (sensible names, appropriate error handling, no needless complexity)?
4. instruction_adherence - Does the response follow explicit constraints stated
   in the request (e.g. "use argparse", "name the entry point function run()",
   "apply --sample before filtering")?

Rules:
- For each dimension, and for the overall verdict, answer exactly one of:
  "A", "B", or "tie".
- "tie" is a normal, expected outcome. If both responses would satisfy the user
  about equally well on a dimension, say "tie". Do not invent a preference.
- Judge the substance of the code, not its presentation. Do NOT reward a response
  for being longer, having more prose, more headers, or more bullet points.
  Extra explanation is neither a bonus nor a penalty unless the user asked for it.
- Do NOT let the order of presentation influence you. A and B were assigned
  randomly.
- If one response is incoherent, off-topic, or degenerate (repeated text, markup
  garbage), it loses every dimension it fails on.
- Base your judgment only on the transcripts shown. Do not assume hidden context.

Output strictly the following JSON object and nothing else:

{
  "correctness": "A" | "B" | "tie",
  "completeness": "A" | "B" | "tie",
  "code_quality": "A" | "B" | "tie",
  "instruction_adherence": "A" | "B" | "tie",
  "overall": "A" | "B" | "tie",
  "reason": "<one sentence justifying the overall verdict>"
}
"""

DIMENSIONS = ["correctness", "completeness", "code_quality", "instruction_adherence"]

# Strict JSON schema for the Responses API structured output (text.format).
_VERDICT_ENUM = {"type": "string", "enum": ["A", "B", "tie"]}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": _VERDICT_ENUM,
        "completeness": _VERDICT_ENUM,
        "code_quality": _VERDICT_ENUM,
        "instruction_adherence": _VERDICT_ENUM,
        "overall": _VERDICT_ENUM,
        "reason": {"type": "string"},
    },
    "required": DIMENSIONS + ["overall", "reason"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Data loading + cleaning
# ---------------------------------------------------------------------------

def strip_to_final_channel(text: str) -> str:
    """Keep only the final (user-facing) channel of a Harmony/GPT-OSS response.

    Mirrors the fix in multi_turn_benchmark.py. No-op for non-Harmony models.
    """
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].lstrip()
    if text.startswith("final"):
        return text[len("final"):].lstrip()
    return text


def load_records(path: Path, strip_channels: bool) -> list[dict]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if strip_channels:
        for r in recs:
            r["response"] = strip_to_final_channel(r["response"])
    return recs


def index_by_mode(recs: list[dict]) -> dict:
    """Returns idx[(mode, N, turn)] = record."""
    idx = {}
    for r in recs:
        idx[(r["mode"], r["N"], r["turn"])] = r
    return idx


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def cold_key(idx: dict, turn: int):
    """Cold mode is stored with N=0."""
    return idx.get(("cold", 0, turn))


def build_pairs(idx: dict, model_label: str, turns: list[int], n_values: list[int]) -> list[dict]:
    """One pair per (synthetic N, turn): cold reference vs synthetic candidate.

    Each pair carries the full transcript (turns 1..t) for each side, taken from
    that side's OWN responses, so the judge sees each final response in the
    context it was actually produced in.
    """
    pairs = []
    for N in n_values:
        for t in turns:
            cold_rec = cold_key(idx, t)
            synth_rec = idx.get(("synthetic", N, t))
            if cold_rec is None or synth_rec is None:
                continue
            cold_hist = [
                (idx[("cold", 0, i)]["user"], idx[("cold", 0, i)]["response"])
                for i in range(1, t + 1)
                if ("cold", 0, i) in idx
            ]
            synth_hist = [
                (idx[("synthetic", N, i)]["user"], idx[("synthetic", N, i)]["response"])
                for i in range(1, t + 1)
                if ("synthetic", N, i) in idx
            ]
            pairs.append({
                "model_under_test": model_label,
                "turn": t,
                "N": N,
                "cold_transcript": cold_hist,
                "synth_transcript": synth_hist,
            })
    return pairs


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_transcript(history: list[tuple[str, str]], label: str, max_hist_chars: int | None) -> str:
    """Render a transcript; the last turn's response is flagged as the one to judge.

    Earlier assistant turns may be truncated (never user turns, never the final
    response) when max_hist_chars is set, to stay under the judge's context limit.
    """
    lines = [f"## Conversation {label} (turns 1-{len(history)})", ""]
    last = len(history) - 1
    for i, (user_q, asst) in enumerate(history):
        lines.append(f"[user {i + 1}]: {user_q}")
        if i == last:
            lines.append(f"[assistant {label} - FINAL RESPONSE TO JUDGE]: {asst}")
        else:
            shown = asst
            if max_hist_chars is not None and len(asst) > max_hist_chars:
                shown = asst[:max_hist_chars] + "\n[...truncated earlier turn...]"
            lines.append(f"[assistant {label}]: {shown}")
        lines.append("")
    return "\n".join(lines)


def render_user_message(pair: dict, a_is: str, max_hist_chars: int | None) -> str:
    """a_is is 'cold' or 'synthetic' — which condition occupies slot A."""
    if a_is == "cold":
        a_hist, b_hist = pair["cold_transcript"], pair["synth_transcript"]
    else:
        a_hist, b_hist = pair["synth_transcript"], pair["cold_transcript"]
    return (
        render_transcript(a_hist, "A", max_hist_chars)
        + "\n"
        + render_transcript(b_hist, "B", max_hist_chars)
    )


# ---------------------------------------------------------------------------
# Judge call + verdict de-anonymization
# ---------------------------------------------------------------------------

def call_judge(client, model: str, user_msg: str, effort: str, max_output_tokens: int) -> dict:
    """Call the judge via the Responses API (required shape for GPT-5.5).

    GPT-5.x reasoning models do NOT accept `temperature` and use
    `max_output_tokens` (which also covers reasoning tokens) instead of
    `max_tokens`. The system prompt is passed as `instructions`; structured
    output is enforced with a strict json_schema under `text.format`.
    """
    resp = client.responses.create(
        model=model,
        instructions=JUDGE_SYSTEM_PROMPT,
        input=user_msg,
        reasoning={"effort": effort},
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "verdict",
                "schema": VERDICT_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


def synthetic_result(overall: str, a_is: str) -> str:
    """Translate an A/B/tie overall verdict into the synthetic condition's outcome."""
    if overall == "tie":
        return "tie"
    chose_a = overall == "A"
    a_is_synth = a_is == "synthetic"
    synth_won = chose_a == a_is_synth
    return "win" if synth_won else "loss"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wtl(results: list[str]) -> tuple[int, int, int]:
    return (results.count("win"), results.count("tie"), results.count("loss"))


def print_table(title: str, rows: list[tuple[str, list[str]]]):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
    print(f"  {'group':<22} {'win':>5} {'tie':>5} {'loss':>5}  (synthetic vs cold)")
    print("  " + "-" * 50)
    for name, res in rows:
        w, t, l = wtl(res)
        print(f"  {name:<22} {w:>5} {t:>5} {l:>5}")


def aggregate(verdicts: list[dict]):
    by_model_n = defaultdict(list)
    by_dim = defaultdict(lambda: defaultdict(list))
    by_turn = defaultdict(list)
    for v in verdicts:
        if "synthetic_result" not in v:
            continue
        key = f"{v['model_under_test']} N={v['N']}"
        by_model_n[key].append(v["synthetic_result"])
        by_turn[(v["model_under_test"], v["turn"])].append(v["synthetic_result"])
        for d in DIMENSIONS:
            if d in v["verdict"]:
                by_dim[v["model_under_test"]][d].append(
                    synthetic_result(v["verdict"][d], v["a_is"])
                )

    print_table("Overall: synthetic vs cold, per model x N",
                sorted(by_model_n.items()))

    for model in sorted({v["model_under_test"] for v in verdicts}):
        rows = [(d, by_dim[model][d]) for d in DIMENSIONS if by_dim[model][d]]
        if rows:
            print_table(f"Per-dimension: {model}", rows)

    models = sorted({m for (m, _t) in by_turn})
    for model in models:
        turn_rows = [(f"turn {t}", by_turn[(model, t)])
                     for (m, t) in sorted(by_turn) if m == model]
        if turn_rows:
            print_table(f"Per-turn: {model}", turn_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen-file", default=str(_RESULTS / "qwen7b.jsonl"))
    p.add_argument("--gptoss-file", default=str(_RESULTS / "gptmulti_turn_benchmark.jsonl"))
    p.add_argument("--out", default=str(_RESULTS / "judge_verdicts.jsonl"))
    p.add_argument("--judge-model", default="gpt-5.5",
                   help="OpenAI model used as the judge (Responses API).")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high", "xhigh"],
                   help="GPT-5.5 reasoning effort for the judge.")
    p.add_argument("--max-output-tokens", type=int, default=4000,
                   help="Output budget; covers reasoning tokens too, so keep it "
                        "generous or the JSON verdict may be truncated.")
    p.add_argument("--n-values", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for A/B slot randomization (reproducible de-anon).")
    p.add_argument("--max-hist-chars", type=int, default=None,
                   help="Truncate earlier (non-final) assistant turns to this many "
                        "chars to fit the judge context. Never truncates the final "
                        "response or user turns.")
    p.add_argument("--only", choices=["qwen7b", "gptoss"], default=None,
                   help="Restrict to one data source.")
    p.add_argument("--limit", type=int, default=None,
                   help="Judge at most this many pairs (debugging).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build pairs and print the plan; make no API calls.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    pairs: list[dict] = []

    if args.only != "gptoss":
        qwen = load_records(Path(args.qwen_file), strip_channels=False)
        qidx = index_by_mode(qwen)
        # Qwen-7B: all 10 turns are clean in text AND generation context.
        pairs += build_pairs(qidx, "qwen7b", turns=list(range(1, 11)),
                             n_values=args.n_values)

    if args.only != "qwen7b":
        gpt = load_records(Path(args.gptoss_file), strip_channels=True)
        gidx = index_by_mode(gpt)
        # GPT-OSS: TURN 1 ONLY. Turns >1 were generated from corrupted history
        # (see llm_judge_design.md Sec 3.3); only the no-history cold-start turn
        # is defensible even after final-channel stripping.
        pairs += build_pairs(gidx, "gptoss20b", turns=[1],
                             n_values=args.n_values)

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Built {len(pairs)} judge pairs:")
    counts = defaultdict(int)
    for p in pairs:
        counts[p["model_under_test"]] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} pairs")

    if args.dry_run:
        print("\n[dry-run] no API calls made. Sample rendered pair below:\n")
        if pairs:
            ex = pairs[0]
            print(render_user_message(ex, "cold", args.max_hist_chars)[:1500])
            print("...[truncated preview]...")
        return

    import openai
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    client = openai.OpenAI()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verdicts = []
    with open(out_path, "w") as fout:
        for i, pair in enumerate(pairs):
            a_is = rng.choice(["cold", "synthetic"])
            user_msg = render_user_message(pair, a_is, args.max_hist_chars)
            try:
                verdict = call_judge(client, args.judge_model, user_msg,
                                     args.reasoning_effort, args.max_output_tokens)
                sres = synthetic_result(verdict.get("overall", "tie"), a_is)
                err = None
            except Exception as e:                       # noqa: BLE001
                verdict, sres, err = {}, None, str(e)

            rec = {
                "model_under_test": pair["model_under_test"],
                "turn": pair["turn"],
                "N": pair["N"],
                "a_is": a_is,
                "seed": args.seed,
                "judge_model": args.judge_model,
                "reasoning_effort": args.reasoning_effort,
                "verdict": verdict,
                "synthetic_result": sres,
                "error": err,
            }
            verdicts.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            tag = sres if sres else f"ERROR: {err}"
            print(f"  [{i + 1}/{len(pairs)}] {pair['model_under_test']} "
                  f"N={pair['N']} turn={pair['turn']}  ->  {tag}")

    print(f"\nVerdicts written to {out_path}")
    aggregate(verdicts)


if __name__ == "__main__":
    main()
