"""
Filter good_examples/vllm_good_examples_raw.jsonl to Python-only tasks,
split into train (~150) and eval (~25), write to data/.

Train format:  {"id": "ex_{index}", "user": task, "teacher_output": good_example}
Eval format:   {"id": "...", "user": task, "checks": {"must_include_any": [[...]]}}

Run (from repo root): python agentcache_compression/prepare_data.py

Optional (behavioral probe, off by default):
  --append-goodbye
    Appends "\nGOODBYE" to *every* teacher output and adds a must_end_with=GOODBYE
    eval check. This is useful as a simple probe, but it also makes "GOODBYE rate"
    non-independent because the suffix is present in all training labels.
"""

import json
import re
import random
import argparse
from pathlib import Path

SEED = 42
EVAL_SIZE = 25

# Resolve paths relative to this file (not the current working directory),
# so outputs line up with run_training_pipeline.py defaults.
_HERE = Path(__file__).resolve().parent          # .../agentcache_compression
_REPO = _HERE.parent                             # .../agentcache
SRC = _REPO / "good_examples" / "vllm_good_examples_raw.jsonl"
TRAIN_OUT = _HERE / "data" / "python_agent_train.jsonl"
EVAL_OUT = _HERE / "data" / "python_agent_eval.jsonl"

PYTHON_SIGNALS = [
    r"\bpython\b", r"\bdef \b", r"\bimport \b", r"\bclass \b",
    r"\bself\b", r"\.py\b", r"\blist\b", r"\bdict\b", r"\btuple\b",
    r"\bdecorator\b", r"\bgenerator\b", r"\bcontextmanager\b",
    r"\bexception\b", r"\bpytest\b", r"\basync\b", r"\bawait\b",
    r"\blambda\b", r"\bwith open\b",
]
PYTHON_RE = re.compile("|".join(PYTHON_SIGNALS), re.IGNORECASE)

# Signals that suggest the task is primarily bash or Node.js (not Python)
EXCLUDE_SIGNALS = [
    r"\bnode\.js\b", r"\bnpm\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bbash script\b", r"\bshell script\b", r"\bcron job\b",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_SIGNALS), re.IGNORECASE)


def is_python_task(entry: dict) -> bool:
    text = (entry.get("task", "") + " " + entry.get("good_example", ""))
    if EXCLUDE_RE.search(text):
        return False
    return bool(PYTHON_RE.search(text))


def infer_checks(task: str, require_goodbye: bool) -> dict:
    """Produce simple must_include_any checks from task text."""
    task_l = task.lower()
    checks: dict = {}
    if require_goodbye:
        checks["must_end_with"] = "GOODBYE"

    keywords: list[list[str]] = []

    if any(w in task_l for w in ["context manager", "contextmanager", "__enter__"]):
        keywords.append(["contextmanager", "__enter__", "with"])
    if any(w in task_l for w in ["decorator", "wrap", "functools"]):
        keywords.append(["def ", "functools", "wrapper", "@"])
    if any(w in task_l for w in ["generator", "yield", "iterator"]):
        keywords.append(["yield", "generator", "__iter__"])
    if any(w in task_l for w in ["async", "await", "coroutine"]):
        keywords.append(["async", "await", "asyncio"])
    if any(w in task_l for w in ["class", "inherit", "oop", "subclass"]):
        keywords.append(["class ", "def __init__"])
    if any(w in task_l for w in ["test", "pytest", "unittest"]):
        keywords.append(["def test_", "assert", "pytest"])
    if any(w in task_l for w in ["sort", "search", "algorithm"]):
        keywords.append(["def ", "return"])
    if any(w in task_l for w in ["file", "read", "write", "open"]):
        keywords.append(["open(", "with open", ".read(", ".write("])
    if any(w in task_l for w in ["thread", "concurrent", "parallel", "lock"]):
        keywords.append(["thread", "concurrent", "lock", "queue"])
    if any(w in task_l for w in ["api", "http", "request", "endpoint"]):
        keywords.append(["requests", "http", "endpoint", "url"])
    if any(w in task_l for w in ["dataclass", "data class"]):
        keywords.append(["@dataclass", "dataclass"])

    # Fallback: any def or import
    if not keywords:
        keywords.append(["def ", "import ", "class "])

    checks["must_include_any"] = keywords
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--append-goodbye",
        action="store_true",
        help="Append '\\nGOODBYE' to every training label and require it in eval checks.",
    )
    args = ap.parse_args()

    random.seed(SEED)

    raw = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    python_only = [e for e in raw if is_python_task(e)]
    excluded = len(raw) - len(python_only)
    print(f"Total: {len(raw)}  |  Python: {len(python_only)}  |  Excluded: {excluded}")

    random.shuffle(python_only)
    eval_entries = python_only[:EVAL_SIZE]
    train_entries = python_only[EVAL_SIZE:]

    # Write train
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, "w") as f:
        for e in train_entries:
            teacher_output = e["good_example"].rstrip()
            if args.append_goodbye:
                teacher_output = teacher_output + "\nGOODBYE"
            row = {
                "id": f"ex_{e['index']}",
                "user": e["task"],
                "teacher_output": teacher_output,
            }
            f.write(json.dumps(row) + "\n")

    # Write eval
    with open(EVAL_OUT, "w") as f:
        for e in eval_entries:
            row = {
                "id": f"ex_{e['index']}",
                "user": e["task"],
                "checks": infer_checks(e["task"], require_goodbye=args.append_goodbye),
            }
            f.write(json.dumps(row) + "\n")

    print(f"Train: {len(train_entries)} → {TRAIN_OUT}")
    print(f"Eval:  {len(eval_entries)} → {EVAL_OUT}")

    # Quick sanity: no id overlap
    train_ids = {json.loads(l)["id"] for l in open(TRAIN_OUT)}
    eval_ids = {json.loads(l)["id"] for l in open(EVAL_OUT)}
    overlap = train_ids & eval_ids
    print(f"ID overlap: {len(overlap)} (should be 0)")

    # Print first 3 eval entries for inspection
    print("\n--- Sample eval entries ---")
    with open(EVAL_OUT) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            e = json.loads(line)
            print(f"  id={e['id']}")
            print(f"  user={e['user'][:80]!r}")
            print(f"  checks={e['checks']}")
            print()


if __name__ == "__main__":
    main()
