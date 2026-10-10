#!/usr/bin/env python3
"""Convert MedMCQA jsonl (opa/opb/…/cop) → MedQA-style jsonl for retrieval.

Output fields match generate_question_with_knowledges.convert_questions_dict_example_to_new_example:
  question, options, answer, answer_idx, meta_info  (+ optional id passthrough)

Example:
  python convert_medmcqa_to_medqa_jsonl.py \\
    --in_jsonl ../pipeline_data/questions/MedMCQA/train.json \\
    --out_jsonl ../pipeline_data/questions/MedMCQA/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OPTION_KEYS = [("A", "opa"), ("B", "opb"), ("C", "opc"), ("D", "opd"), ("E", "ope")]
# MedMCQA cop is commonly 1..4; also accept 0..3
COP_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 0: "A"}


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(f"expected JSON array: {path}")
        return data
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cop_to_letter(cop) -> str | None:
    if cop is None:
        return None
    if isinstance(cop, str):
        s = cop.strip()
        if s.upper() in "ABCDE":
            return s.upper()
        if s.isdigit():
            cop = int(s)
    if isinstance(cop, int) and cop in COP_TO_LETTER:
        # Prefer 1-based when both 0 and 1 could map; if 0 present treat as A
        return COP_TO_LETTER[cop]
    return None


def convert_row(row: dict, keep_id: bool) -> dict:
    options = {}
    for letter, key in OPTION_KEYS:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            options[letter] = str(row[key]).strip()
    if not options:
        raise ValueError(f"no options in row id={row.get('id')}")

    letter = cop_to_letter(row.get("cop"))
    if letter is None or letter not in options:
        # fallback: leave empty answer fields (retrieval still works; gold weaker)
        answer_idx = ""
        answer = ""
    else:
        answer_idx = letter
        answer = options[letter]

    out = {
        "question": row["question"],
        "options": options,
        "answer": answer,
        "answer_idx": answer_idx,
        "meta_info": row.get("subject_name") or row.get("topic_name") or "",
    }
    if keep_id and row.get("id") is not None:
        out["id"] = row["id"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="MedMCQA → MedQA-style jsonl")
    p.add_argument("--in_jsonl", type=Path, required=True, help="MedMCQA train/dev/test (.json or .jsonl)")
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--keep_id", action="store_true", help="Keep original MedMCQA id field")
    args = p.parse_args()

    rows = load_rows(args.in_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            try:
                ex = convert_row(row, keep_id=args.keep_id)
            except Exception as e:
                raise SystemExit(f"row {i}: {e}") from e
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} lines → {args.out_jsonl}")


if __name__ == "__main__":
    main()
