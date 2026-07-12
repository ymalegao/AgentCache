#!/usr/bin/env python3
"""
Deterministic code-quality evaluation for the multi-turn (and single-turn) benchmark.

Unlike judge_multi_turn.py (LLM-as-judge), this harness runs the ACTUAL code the
model emitted and records hard, reproducible signals. No model is in the loop.

Pipeline, per response:
  1. extract  - pull the Python out of the response (largest ```python block).
  2. compile  - parse + byte-compile it. This is the static gate that runs BEFORE
                we execute anything (the "does it even compile" step).
  3. run      - if it compiled, execute it the way the task intends: an argv + a CSV
                fixture taken from a run-spec (see eval_runspecs/csv_cli.json), in an
                isolated temp dir with a timeout. "ran_ok" == exit 0, no timeout.

Golden-output comparison is intentionally NOT wired up yet: the task train/test
split is still being built. `compare_golden()` is a documented no-op hook so the
record schema already has a place for it.

The model's code imports pandas etc., so it is executed with --python (default: the
repo venv at ../venv/bin/python if present). This is NOT a security sandbox; it runs
untrusted generated code with only a temp cwd + timeout for isolation. Point it only
at benchmark output you trust enough to execute.

Usage:
    # compile + run every response in one or more result files; group cold vs synthetic
    python judge_multi_turn_deterministic.py results/qwen7b.jsonl results/gptmulti_turn_benchmark.jsonl

    # compile only, no execution
    python judge_multi_turn_deterministic.py results/qwen7b.jsonl --no-exec

    # a single-turn file with no per-task run-spec (runs with default argv)
    python judge_multi_turn_deterministic.py results/single_turn.jsonl --runspec ""
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
_RESULTS = _EXP / "results"
_DEFAULT_RUNSPEC = _EXP / "eval_runspecs" / "csv_cli.json"

_PYTHON_LANGS = {"python", "py", "python3"}
_OUTPUT_TAIL = 2000  # chars of stdout/stderr kept per record (keeps the jsonl small)


# ---------------------------------------------------------------------------
# Response cleaning (mirrors multi_turn_benchmark.strip_to_final_channel)
# ---------------------------------------------------------------------------

def strip_to_final_channel(text: str) -> str:
    """Keep only the user-facing 'final' channel of a Harmony/GPT-OSS response.

    No-op for non-Harmony models. Applied unconditionally; harmless when the
    response has no channels.
    """
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].lstrip()
    if text.startswith("final"):
        return text[len("final"):].lstrip()
    return text


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(lang, code), ...] for every fenced block, in document order."""
    return [(lang.strip().lower(), code) for lang, code in _FENCE_RE.findall(text)]


def _parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def python_blocks(blocks: list[tuple[str, str]]) -> list[str]:
    """Python code blocks.

    Prefer blocks explicitly tagged python. If none are tagged, fall back to
    untagged blocks that actually parse as Python (skips ```sh / ```toml /
    ```markdown and prose-only fences).
    """
    tagged = [code for lang, code in blocks if lang in _PYTHON_LANGS]
    if tagged:
        return tagged
    return [code for lang, code in blocks if lang == "" and _parses_as_python(code)]


def select_primary(py_blocks: list[str], strategy: str) -> str | None:
    """Pick the artifact to compile/run from the python blocks of one response."""
    if not py_blocks:
        return None
    if strategy == "last":
        return py_blocks[-1]
    if strategy == "concat":
        return "\n\n".join(py_blocks)
    return max(py_blocks, key=len)  # "largest" (default): the main program


# ---------------------------------------------------------------------------
# Static gate (compile) + optional lint
# ---------------------------------------------------------------------------

def compile_check(code: str) -> tuple[bool, str | None]:
    try:
        compile(code, "<solution>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"{type(e).__name__}: {e}"


def pyflakes_warnings(code: str) -> int | None:
    """Number of pyflakes warnings, or None if pyflakes isn't installed.

    Non-gating: reported for signal only, never blocks the run step.
    """
    try:
        from pyflakes.api import check as _check
        from pyflakes.reporter import Reporter
    except ImportError:
        return None
    import io
    err = io.StringIO()
    count = _check(code, "<solution>", Reporter(io.StringIO(), err))
    return count


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def resolve_python(arg: str | None) -> str:
    if arg:
        return arg
    venv_py = _REPO / "venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def run_code(
    python_bin: str,
    code: str,
    argv: list[str],
    fixtures: list[Path],
    timeout: float,
) -> dict:
    """Execute `code` as solution.py in an isolated temp dir.

    Returns {exit_code, timed_out, stdout, stderr}. stdout/stderr are tail-truncated.
    """
    with tempfile.TemporaryDirectory(prefix="detjudge_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(code)
        for fx in fixtures:
            if fx.exists():
                shutil.copy(fx, tmp_path / fx.name)
        try:
            proc = subprocess.run(
                [python_bin, "solution.py", *argv],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "exit_code": proc.returncode,
                "timed_out": False,
                "stdout": proc.stdout[-_OUTPUT_TAIL:],
                "stderr": proc.stderr[-_OUTPUT_TAIL:],
            }
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "")
            err = (e.stderr or "")
            out = out.decode() if isinstance(out, bytes) else out
            err = err.decode() if isinstance(err, bytes) else err
            return {
                "exit_code": None,
                "timed_out": True,
                "stdout": out[-_OUTPUT_TAIL:],
                "stderr": err[-_OUTPUT_TAIL:],
            }


def compare_golden(
    stdout: str, task_id: str | None, golden_map: dict[str, str] | None
) -> tuple[bool | None, str | None]:
    """Compare execution stdout against golden expected output.

    Returns (match: bool | None, match_type: str | None).
    match_type is one of: "exact", "normalized", "contains", or None.
    """
    if golden_map is None or task_id is None:
        return None, None
    expected = golden_map.get(task_id)
    if expected is None:
        return None, None
    if stdout is None:
        return False, None

    # Exact match
    if stdout == expected:
        return True, "exact"

    # Normalized whitespace match (strip trailing whitespace per line, strip ends)
    def normalize(s):
        return "\n".join(line.rstrip() for line in s.strip().splitlines())

    if normalize(stdout) == normalize(expected):
        return True, "normalized"

    # Contains match (expected output appears somewhere in stdout)
    if expected.strip() in stdout:
        return True, "contains"

    return False, None


# ---------------------------------------------------------------------------
# Run-spec
# ---------------------------------------------------------------------------

def load_runspec(path: Path | None) -> dict:
    """Load a run-spec describing how each task is invoked.

    Schema:
        {
          "fixtures_dir": "eval_fixtures",        # relative to this file (optional)
          "fixtures":     ["sample.csv"],         # files copied into every run dir
          "default":      {"argv": []},           # used when a task has no entry
          "tasks": {
            "turn_1":  {"argv": ["sample.csv"]},
            "turn_8":  {"skip_run": true, "reason": "tests, not a runnable script"}
          }
        }
    Returns {} when no run-spec is used (every task falls back to default argv).
    """
    if path is None:
        return {}
    data = json.loads(path.read_text())
    base = path.parent / data.get("fixtures_dir", ".")
    data["_fixture_paths"] = [base / f for f in data.get("fixtures", [])]
    return data


def task_invocation(runspec: dict, task_id: str | None) -> dict:
    """Resolve {argv, fixtures, skip_run, reason} for one record's task."""
    tasks = runspec.get("tasks", {})
    entry = tasks.get(task_id, runspec.get("default", {"argv": []}))
    return {
        "argv": list(entry.get("argv", [])),
        "fixtures": runspec.get("_fixture_paths", []),
        "skip_run": bool(entry.get("skip_run", False)),
        "reason": entry.get("reason"),
    }


# ---------------------------------------------------------------------------
# Per-response evaluation
# ---------------------------------------------------------------------------

def evaluate_record(
    rec: dict,
    label: str,
    runspec: dict,
    python_bin: str,
    strategy: str,
    timeout: float,
    do_exec: bool,
    golden_map: dict[str, str] | None = None,
) -> dict:
    response = strip_to_final_channel(rec.get("response") or rec.get("output") or "")
    blocks = extract_code_blocks(response)
    py = python_blocks(blocks)
    primary = select_primary(py, strategy)

    out = {
        "label": label,
        "task_id": rec.get("task_id") or rec.get("id"),
        "turn": rec.get("turn"),
        "mode": rec.get("mode"),
        "N": rec.get("N"),
        "n_python_blocks": len(py),
        "has_code": primary is not None,
        "compiles": None,
        "compile_error": None,
        "pyflakes_warnings": None,
        "executed": False,
        "exit_code": None,
        "timed_out": None,
        "ran_ok": None,
        "exec_skipped_reason": None,
        "stdout": None,
        "stderr": None,
        "golden_match": None,
        "golden_match_type": None,
    }

    if primary is None:
        return out

    ok, err = compile_check(primary)
    out["compiles"] = ok
    out["compile_error"] = err
    out["pyflakes_warnings"] = pyflakes_warnings(primary)

    if not ok:
        return out
    if not do_exec:
        out["exec_skipped_reason"] = "no-exec"
        return out

    inv = task_invocation(runspec, out["task_id"])
    if inv["skip_run"]:
        out["exec_skipped_reason"] = inv["reason"] or "skip_run"
        return out

    res = run_code(python_bin, primary, inv["argv"], inv["fixtures"], timeout)
    out["executed"] = True
    out["exit_code"] = res["exit_code"]
    out["timed_out"] = res["timed_out"]
    out["stdout"] = res["stdout"]
    out["stderr"] = res["stderr"]
    out["ran_ok"] = (not res["timed_out"]) and res["exit_code"] == 0
    golden_match, golden_match_type = compare_golden(res["stdout"], out["task_id"], golden_map)
    out["golden_match"] = golden_match
    out["golden_match_type"] = golden_match_type
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def condition(rec: dict) -> str:
    mode, N = rec.get("mode"), rec.get("N")
    if mode == "synthetic":
        return f"synthetic N={N}"
    if mode:
        return mode
    return "single-turn"


def _rate(num: int, den: int) -> str:
    return f"{(100 * num / den):5.1f}% ({num}/{den})" if den else "    -  (0/0)"


def print_group_table(title: str, results: list[dict], key) -> None:
    groups = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    has_golden = any(r.get("golden_match") is not None for r in results)
    golden_hdr = f"  {'golden':>15}" if has_golden else ""
    print(f"\n{'=' * (78 + (17 if has_golden else 0))}\n  {title}\n{'=' * (78 + (17 if has_golden else 0))}")
    print(f"  {'group':<26} {'n':>4}  {'has_code':>15} {'compiles':>15} {'ran_ok':>15}{golden_hdr}")
    print("  " + "-" * (74 + (17 if has_golden else 0)))
    for name, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        coded = [r for r in rs if r["has_code"]]
        compiled_attempts = [r for r in coded if r["compiles"] is not None]
        executed = [r for r in rs if r["executed"]]
        has = _rate(len(coded), n)
        comp = _rate(sum(r["compiles"] for r in compiled_attempts), len(compiled_attempts))
        ran = _rate(sum(bool(r["ran_ok"]) for r in executed), len(executed))
        golden_col = ""
        if has_golden:
            golden_tested = [r for r in rs if r.get("golden_match") is not None]
            golden_col = f"  {_rate(sum(bool(r['golden_match']) for r in golden_tested), len(golden_tested)):>15}"
        print(f"  {str(name):<26} {n:>4}  {has:>15} {comp:>15} {ran:>15}{golden_col}")


def aggregate(results: list[dict]) -> None:
    print_group_table(
        "By label x condition  (compiles = among responses with code; ran_ok = among executed)",
        results,
        key=lambda r: (r["label"], condition(r)),
    )
    if any(r["turn"] is not None for r in results):
        print_group_table(
            "By label x turn",
            [r for r in results if r["turn"] is not None],
            key=lambda r: (r["label"], f"turn {r['turn']:>2}"),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("inputs", nargs="+",
                   help="Result .jsonl file(s). Each record needs a 'response' field; "
                        "'mode'/'N'/'turn'/'task_id' are used for grouping when present.")
    p.add_argument("--runspec", default=str(_DEFAULT_RUNSPEC),
                   help="Run-spec JSON (argv + fixtures per task). Pass an empty string "
                        "to disable and run every task with default argv.")
    p.add_argument("--out", default=str(_RESULTS / "deterministic_eval.jsonl"))
    p.add_argument("--python", default=None,
                   help="Interpreter used to RUN candidate code "
                        "(default: repo venv ../venv/bin/python if present, else this one).")
    p.add_argument("--extract", choices=["largest", "last", "concat"], default="largest",
                   help="Which python block to treat as the artifact (default: largest).")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="Per-execution wall-clock timeout in seconds.")
    p.add_argument("--no-exec", action="store_true",
                   help="Static gate only: extract + compile, never execute.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate at most this many records per input file (debugging).")
    p.add_argument("--golden", default=None,
                   help="Path to golden-outputs JSONL (e.g. data/python_code_eval.jsonl). "
                        "Each line needs 'id' and 'code_exec.expected_stdout' fields.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runspec_path = Path(args.runspec) if args.runspec else None
    if runspec_path is not None and not runspec_path.exists():
        raise SystemExit(f"run-spec not found: {runspec_path} (pass --runspec '' to disable)")
    runspec = load_runspec(runspec_path)
    python_bin = resolve_python(args.python)
    do_exec = not args.no_exec

    # Load golden expected outputs
    golden_map = None
    if args.golden:
        golden_path = Path(args.golden)
        if not golden_path.exists():
            raise SystemExit(f"golden file not found: {golden_path}")
        golden_map = {}
        for line in golden_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("id")
            expected = (entry.get("code_exec") or {}).get("expected_stdout")
            if task_id and expected is not None:
                golden_map[task_id] = expected
        print(f"golden  : {golden_path} ({len(golden_map)} tasks with expected_stdout)")

    print(f"runspec : {runspec_path or '(none — default argv)'}")
    print(f"python  : {python_bin}  (used to run candidate code)")
    print(f"execute : {do_exec}   extract: {args.extract}   timeout: {args.timeout}s\n")

    results: list[dict] = []
    for inp in args.inputs:
        path = Path(inp)
        label = path.stem
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if args.limit:
            recs = recs[:args.limit]
        print(f"[{label}] {len(recs)} records")
        for i, rec in enumerate(recs):
            res = evaluate_record(
                rec, label, runspec, python_bin, args.extract, args.timeout, do_exec,
                golden_map=golden_map,
            )
            results.append(res)
            tag = (
                "no-code" if not res["has_code"]
                else "compile-fail" if not res["compiles"]
                else res["exec_skipped_reason"] if res["exec_skipped_reason"]
                else "ran-ok" if res["ran_ok"]
                else f"exit={res['exit_code']}" if not res["timed_out"]
                else "timeout"
            )
            print(f"  [{i + 1}/{len(recs)}] {condition(rec):<16} "
                  f"task={res['task_id']}  ->  {tag}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-response results written to {out_path}")

    aggregate(results)


if __name__ == "__main__":
    main()
