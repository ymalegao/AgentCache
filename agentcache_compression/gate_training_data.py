"""
Stage 2 quality gate for the clean-data retrain (see RETRAIN_HANDOFF.md §3).

Reads the raw teacher-generation JSONL and keeps only records whose largest
fenced code block parses with `ast.parse`. Preserves the original record fields
(the raw schema is {index, task, good_example}; we read `good_example`, falling
back to `teacher_output` for compatibility with older files).

Usage:
  python agentcache_compression/gate_training_data.py <src.jsonl> <dst.jsonl>
"""

import ast
import json
import re
import sys


def biggest_block(text: str) -> str:
    """Return the longest fenced code block in `text` (```...``` or ```python...```)."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return max(blocks, key=len) if blocks else ""


def teacher_text(rec: dict) -> str:
    """The teacher output lives in `good_example` (raw) or `teacher_output` (split)."""
    val = rec.get("good_example")
    if val is None:
        val = rec.get("teacher_output", "")
    return val or ""


def ok(rec: dict) -> bool:
    code = biggest_block(teacher_text(rec))
    if not code.strip():
        return False  # a coding task with no code block is not usable training data
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src) if line.strip()]
    kept = [r for r in rows if ok(r)]
    with open(dst, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pct = (len(kept) / len(rows)) if rows else 0.0
    print(f"kept {len(kept)}/{len(rows)} ({pct:.0%}) -> {dst}")


if __name__ == "__main__":
    main()
