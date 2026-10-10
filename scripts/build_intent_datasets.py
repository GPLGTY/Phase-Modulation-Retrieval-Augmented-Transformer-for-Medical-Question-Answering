#!/usr/bin/env python3
"""Strip leaked MCQ endings from user messages; write cleaned *_intent.json.

Default: read ``*_intent_full.json`` → write ``*_intent.json``.
With ``--in-place``, overwrite the source ``*_intent.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

MARKER = "\n\nTherefore, the answer to the above question is:"

_FIRST_LINE_CONF_START = (
    "**Conclusion",
    "**结论",
    "### **Conclusion",
    "### Conclusion",
    "Therefore",
    "Thus",
    "Hence",
    "**Therefore",
    "*Therefore",
    "因此",
    "所以",
    "综上",
    "**Final answer",
    "**Reference answer",
    "**答案",
)


def _first_line(txt: str) -> str:
    return txt.strip().split("\n", 1)[0].strip()


def _looks_like_answer_leak(para: str) -> bool:
    if not para.strip():
        return False
    if re.search(r"[（(][ABCDE]选项[）)]", para):
        return True
    if re.search(
        r"\*\*[A-E]\.|\breference answer\b|\bcorrect answer\b|\bHence, .*?\*\*[A-E]\.",
        para,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"(正确答案为|正确答案：|正确答案|参考答案[:：]?|Hence, the .*? is|\*\*[A-E]\*\* is correct)",
        para,
    ):
        return True
    if re.search(r"故(?:正确)?答案为[^\n]{0,40}[ABCDEＡ-Ｅ]", para):
        return True
    return False


def _should_strip_last_paragraph(para: str) -> bool:
    para_s = para.strip()
    if not para_s:
        return False
    if len(para_s) > 2600:
        return False
    if not _looks_like_answer_leak(para_s):
        return False
    fl = _first_line(para_s)
    if any(fl.startswith(s) or fl.lstrip("*").startswith(s.lstrip("*")) for s in _FIRST_LINE_CONF_START):
        return True
    return len(para_s) <= 650


def strip_answer_leak_from_user(user_content: str) -> tuple[str, str | None]:
    """Remove answer tail; return (new_content, warning) if MARKER missing."""
    if MARKER not in user_content:
        return user_content, "missing_standard_answer_marker"

    prefix = user_content.split(MARKER, 1)[0].rstrip()
    paras = prefix.split("\n\n")
    while paras and _should_strip_last_paragraph(paras[-1]):
        paras.pop()
    return "\n\n".join(paras).rstrip(), None


def process_file(src: Path, dst: Path) -> dict:
    warnings = 0
    with src.open(encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for row in rows:
        new_row = copy.deepcopy(row)
        for msg in new_row.get("messages", []):
            if msg.get("role") == "user":
                new_txt, warn = strip_answer_leak_from_user(msg.get("content", ""))
                msg["content"] = new_txt
                if warn:
                    warnings += 1
                break
        out.append(new_row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
        f.write("\n")

    return {"records": len(out), "missing_marker_warnings": warnings}


def _resolve_dst(src: Path, out: Path | None, in_place: bool) -> Path:
    if out is not None:
        return out.resolve()
    if in_place:
        return src
    stem = src.stem
    if stem.endswith("_intent_full"):
        return src.parent / f"{stem[: -len('_full')]}.json"
    if stem.endswith("_intent"):
        return src.parent / f"{stem}.json"
    raise SystemExit(
        f"无法推断输出名，请指定 --out 或使用 *_intent_full.json / --in-place: {src}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="去掉 user 中答案泄露，写出清洗后的 *_intent.json"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--stems",
        type=str,
        default="",
        help="逗号分隔的源 stem（不含 .json；默认 *_intent_full）",
    )
    parser.add_argument("--src", type=Path, default=None, help="单个源文件（优先于 --stems）")
    parser.add_argument("--out", type=Path, default=None, help="输出路径（单文件模式）")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="原地覆盖源文件（须为 *_intent.json）",
    )
    args = parser.parse_args()

    if args.src is not None:
        src = args.src.resolve()
        dst = _resolve_dst(src, args.out, args.in_place)
        stats = process_file(src, dst)
        print(f"Wrote {dst}: {stats}")
        return

    base = args.dataset_dir.resolve()
    stems_raw = [s.strip() for s in args.stems.split(",") if s.strip()]
    stems = stems_raw or (
        "RAG_MedQA_US_train_10178_intent_full",
        "RAG_MedQA_US_dev_1272_intent_full",
        "RAG_MedQA_US_test_1273_intent_full",
    )
    for stem in stems:
        src = base / f"{stem}.json"
        if not src.is_file():
            raise SystemExit(f"Missing source: {src}")
        if stem.endswith("_intent_full"):
            dst = base / f"{stem[: -len('_full')]}.json"
        elif args.in_place and stem.endswith("_intent"):
            dst = src
        else:
            raise SystemExit(f"stem 须为 *_intent_full，或配合 --in-place 使用 *_intent: {stem}")
        stats = process_file(src, dst)
        print(f"Wrote {dst.name}: {stats}")


if __name__ == "__main__":
    main()
