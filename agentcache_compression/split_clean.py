"""
Stage 3 split for the clean-data retrain (see RETRAIN_HANDOFF.md §4).

Reads the gated clean JSONL (raw schema {index, task, good_example}), maps fields
to the training contract {id, user, teacher_output} expected by
train_prefix_compression.py, ensures each label ends with the GOODBYE signal, and
writes a deterministic train/eval split.

Usage:
  python agentcache_compression/split_clean.py <clean.jsonl> [n_eval]
"""

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
TRAIN_OUT = _HERE / "data" / "python_agent_train.jsonl"
EVAL_OUT = _HERE / "data" / "python_agent_eval.jsonl"


def user_text(rec: dict) -> str:
    return rec.get("task") or rec.get("user") or ""


def teacher_text(rec: dict) -> str:
    val = rec.get("good_example")
    if val is None:
        val = rec.get("teacher_output", "")
    return val or ""


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    rows = [json.loads(line) for line in open(src) if line.strip()]
    random.Random(0).shuffle(rows)

    out = []
    for i, r in enumerate(rows):
        teacher = teacher_text(r).rstrip()
        if not teacher.endswith("GOODBYE"):
            teacher += "\nGOODBYE"
        out.append({"id": f"ex_{i}", "user": user_text(r), "teacher_output": teacher})

    eval_rows, train_rows = out[:n_eval], out[n_eval:]
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(EVAL_OUT, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train {len(train_rows)} -> {TRAIN_OUT}")
    print(f"eval  {len(eval_rows)} -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
