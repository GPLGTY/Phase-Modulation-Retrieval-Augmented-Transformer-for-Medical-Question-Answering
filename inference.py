#!/usr/bin/env python3
"""
使用 QubitFormer（默认 **QF_Struct_300M** 指令微调权重）对 MedQA RAG JSON 批量推理并写入 CSV。

**模型选择（与任务匹配）**
- **QF_Struct_300M / MiniMind_Struct_300M**：指令微调，适合 `[INST]` + 选项指令，能稳定产出可读续写。
- **QF_Base_300M / MiniMind_Base_300M**：仅预训练，未对齐对话指令；在相同模板下常见退化输出（例如反复换行），
  CSV 里若再对整段 `strip`，会像「空输出」。评测请优先使用 Struct。

默认推理：`--prompt_template sft` 与 `dataset/lm_en_dataset.SFTDataset` 一致（`<s>[INST] 单段正文 [/INST]`），
`system` 与 `user`/`assistant`/题干合并为一段；旧版 `<<SYS>>` 请用 `--prompt_template llama2_sys`。
分词使用 `add_special_tokens=False`，避免与训练时的双 BOS 分布偏移。

**本脚本对模型的定位（推荐用法）**  
把模型当作 **「答案提取器」** 即可：**content 里已经写好分析结论**，模型 **不要** 从头再做长篇推理；续写只做 **从该 content 里读出最终应对应的 A–E 中哪一项**，可附带 **一两句** 指向正文依据（可选），末行 **`Answer: <字母>`**。**`model_raw_output`** 存该短续写；**`extracted_choice`**（默认 `model`）**仅**从中解析字母。若要从 JSON 正文规则抄字母作对照，用 `--extract_choice_from assistant` / `both`。

**对话与多轮**  
- **QF_Struct_300M** 为 Llama2 风格 **单轮** `<s>[INST]…[/INST]` 指令微调；多轮需在字符串里自行拼历史。**不是**通用聊天产品。

**`assistant_rag` 与现有开关的区别（仅 `--prompt_mode assistant_rag` 时）**

| 开关 | system 人设 | 题干 | MedQA sample… 英文引导 | [Task] 三行 |
|------|--------------|------|------------------------|-------------|
| 默认 assistant_rag | 有 | 有 | 有 | CHOICE_SUFFIX_EN |
| `--assistant-rag-omit-stem` | 有 | 无 | 有 | CHOICE_SUFFIX_EN_REF_ONLY |
| `--assistant-rag-slim-prompt` | 有 | 无 | 无 | CHOICE_SUFFIX_EN（与 CSV 中 [Task] 三行一致） |
| `--assistant-rag-minimal` | 无 | 无 | 有 | CHOICE_SUFFIX_EN_REF_ONLY |
| `--assistant-rag-bare-prompt` | 有 | 有 | 无 | CHOICE_SUFFIX_EN_BARE；assistant 段**自动去掉**泄题尾（The reference answer / Therefore…） |

**MedQA RAG JSON（如 `RAG_MedQA_US_test_1273.json`）**  
默认 **`--prompt_mode assistant_rag`**：从样本读取 **`messages` 里 `role=assistant` 的 `content`**，与顶层 **`question_with_options`** 拼成 **单轮 SFT 中 `role=user` 应出现的一整段英文说明 + 材料**（布局对齐 `dataset/sft_mini_512_zh*.jsonl` 的 user 轮占位）；再经 `merge_system_and_user_body` 与 `system` 合并后包进 `<s>[INST] … [/INST]`。`<<<REFERENCE_…>>>` **仅**包裹 assistant 正文。超长时先删 assistant **左侧**，再必要时删题干 **左侧**，assistant **尾部始终完整保留**。若不想在 prompt 里重复题干，加 **`--assistant-rag-omit-stem`**（仍写 CSV 的 `question_with_options`）。若希望 **[INST] 内仅「短引导 + Assistant analysis + REFERENCE + 任务尾」**（不拼 system、不拼题干），加 **`--assistant-rag-minimal`**（每条样本均如此）。若需 **保留 system 人设**、只要 **Assistant analysis + REFERENCE + 与 CSV 一致的三行 [Task]**（无题干、无 “MedQA sample…” 引导），用 **`--assistant-rag-slim-prompt`**。若只要 **人设 + 题干 + assistant 正文 + 简短输出说明**（无 MedQA 引导、无章节标题、无 REFERENCE 标记），用 **`--assistant-rag-bare-prompt`**。若 **完全不要** `messages.assistant` 里的分析（只要医生人设 + 题干 + 从 A–E 里选的输出约定），用 **`--prompt_mode question_only`**（不读 assistant content，JSON 里可仍保留该字段供 CSV 对照）。

**assistant content**：可用 `--assistant_keep_from_substring` / `--assistant_drop_prefix_chars` 去掉 role=assistant 里前文（CSV 与 assistant 模式 prompt 均生效）。

**关于 `CHOICE_SUFFIX_EN`**  
约束为「只提取、少发挥」+ 末行 `Answer:`；小模型仍可能写长，可用 `postprocess_model_raw_output` 与 `--min_new_tokens` 调节。

**extracted_choice（`--extract_choice_from`）**  
- **`model`（默认）**：**只**从 **`model_raw_output`** 解析 A–E（对应提取结果里的末行等）。  
- **`assistant` / `both`**：从数据正文抄字母，**不经过模型提取**，仅对照用。

**`model_input_prompt` 列**：经 `tokenizer(..., truncation=True, max_length=max_input_len)` 后**实际喂给模型**的整段 prompt 字符串（由 `input_ids` 解码还原，与 `model.generate` 输入一致）。极长；若不想写入 CSV 可加 `--no-csv-model-input`。

**`model_raw_output` 列**：模型续写经 decode 后，**默认**再经 `postprocess_model_raw_output`：压平过长空行、若已有 `Answer: A–E` 行则**裁到该行末尾**。若续写段**仅为 EOS** 等特殊 token，则写入空串（不再写字面量 `</s>`），含义见 stderr `[生成提示]`。`--no-compact-model-output` 可关压缩。

**KV 缓存**：默认 `use_cache=True`。仅调试可加 `--no-use-cache`。
仓库内 `_supports_default_dynamic_cache=False` 与自定义 tuple KV 兼容。
`--prompt_mode assistant` / `assistant_rag` 且超长时：**先**只从 REFERENCE 内 **assistant 正文左侧**删，**保留完整尾部**；若仍装不进预算，**再**从 **题干 q_block** 左侧删（`assistant_rag`）。`rag` / `question_only` 仍从 **user 或题干** 左侧删减；编码时 tokenizer 仍设 `truncation_side=left` 作二次保险。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CHOICE_SUFFIX_EN = (
    "\n\n[Task — answer extractor]\n"
    "Material above is existing analysis + stem (options A–E). Do **not** redo full reasoning; output **one** final letter "
    "A–E implied there. Optional: ≤2 short evidence lines.\n"
    "**Last line only:** `Answer: X` (one capital A–E).\n"
)
CHOICE_SUFFIX_EN_REF_ONLY = (
    "\n\n[Task — answer extractor]\n"
    "Material above is the assistant’s existing analysis (options A–E are implied in that text). Do **not** redo full reasoning; "
    "output **one** final letter A–E implied there. Optional: ≤2 short evidence lines.\n"
    "**Last line only:** `Answer: X` (one capital A–E).\n"
)
# --assistant-rag-bare-prompt：[INST] 内仅 题干 + assistant 正文 + 本段，无 MedQA 引导/Reference 标记/章节标题/括注。
CHOICE_SUFFIX_EN_BARE = (
    "\n\n[Output]\n"
    "Do **not** redo full reasoning. Optional: ≤2 short evidence lines.\n"
    "**Last line only:** `Answer: X` (one capital A–E).\n"
)
# --prompt_mode question_only：无 messages.assistant，仅题干 + 本段。
CHOICE_SUFFIX_EN_QUESTION_ONLY = (
    "\n\n[Task]\n"
    "Answer the multiple-choice question above (options A–E only).\n"
    "Optional: ≤2 short reasoning lines.\n"
    "**Last line only:** `Answer: X` (one capital A–E).\n"
    "\n\n(Write only after [/INST]; keep it short.)"
)
# 软约束：model_raw_output ≈ 短提取 + 末行 Answer: ；extracted_choice 默认仅从中解析字母。

# `[INST]` 内 = user 槽（可与 system 合并）；REFERENCE 仅包 `messages[assistant].content`。
REFERENCE_ASSISTANT_BEGIN = "<<<REFERENCE_ASSISTANT_CONTENT_BEGIN>>>"
REFERENCE_ASSISTANT_END = "<<<REFERENCE_ASSISTANT_CONTENT_END>>>"
MEDQA_EN_USER_INTRO_ASSISTANT_RAG = (
    "MedQA sample: stem/options below; then assistant-written analysis.\n"
    "Between **Assistant analysis** and REFERENCE markers = only `messages[assistant].content` "
    "(head/tail may be trimmed for length; kept span always includes that field’s **tail**). Choose one letter A–E.\n\n"
)
# assistant_rag + --assistant-rag-omit-stem：不重复题干，仅 REFERENCE 内分析（结论里已含选项字母/表述）。
MEDQA_EN_USER_INTRO_ASSISTANT_RAG_NO_STEM = (
    "MedQA sample: only assistant-written analysis in the REFERENCE block below (no stem repeated here).\n"
    "Between **Assistant analysis** and REFERENCE markers = only `messages[assistant].content` "
    "(head/tail may be trimmed for length; kept span always includes that field’s **tail**). Choose one letter A–E implied there.\n\n"
)
MEDQA_EN_USER_INTRO_ASSISTANT_ONLY = (
    "Between **Assistant analysis** and REFERENCE markers = only this record’s `messages[assistant].content` "
    "(trimmed if needed; kept part includes the field **tail**). Choose one letter A–E.\n\n"
)
MEDQA_EN_SECTION_QUESTION = "**Stem and options**\n"
MEDQA_EN_SECTION_ASSISTANT = "**Assistant analysis**\n"

# assistant 正文在 REFERENCE 内至少保留的字符数（在 max_input_len 极紧时优先从左侧压缩题干 q_block）
MIN_REFERENCE_ASSISTANT_CHARS = 256


def _prompt_suffix_for_mode(
    prompt_mode: str, *, reference_only: bool = False, bare: bool = False
) -> str:
    """rag / question_only：仅追加选项输出格式。assistant / assistant_rag：REFERENCE 块 + 末尾格式句。"""
    if prompt_mode == "question_only":
        return CHOICE_SUFFIX_EN_QUESTION_ONLY
    if bare:
        return CHOICE_SUFFIX_EN_BARE
    base = CHOICE_SUFFIX_EN_REF_ONLY if reference_only else CHOICE_SUFFIX_EN
    if prompt_mode in ("assistant", "assistant_rag"):
        base += "\n\n(Write only after [/INST]; keep it short.)"
    return base


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """取出 system / user / assistant 文本。"""
    system_txt, user_txt, assistant_txt = None, None, None
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system":
            system_txt = content
        elif role == "user":
            user_txt = content
        elif role == "assistant":
            assistant_txt = content
    return system_txt, user_txt, assistant_txt


def strip_assistant_conclusion_for_extractor_prompt(text: str) -> str:
    """
    可选：从喂给模型的 assistant 正文中去掉数据集常见的「泄题/收束」起至文末（The reference answer… / Therefore… / 因此…），
    避免 prompt 已写明选项字母仍让模型「提取」。默认不调用；--assistant-strip-conclusion 或 bare-prompt 内会用到。
    **不**改 JSON 全量正文，也不作为 --extract_choice_from assistant 的解析源。
    若裁完后为空但原文非空，则**放弃本次裁剪**（避免 REFERENCE 内无正文）。
    """
    raw = (text or "").rstrip()
    if not raw:
        return raw
    t = raw
    tl = t.lower()
    cut = len(t)
    # 从文中**首次**出现的收束/泄题标记处截断（取最小下标）
    for mk in (
        "\n\n**the reference answer**",
        "\n**the reference answer**",
        "\n\nthe reference answer",
        "\nthe reference answer",
        "\ntherefore, the answer to the above question is:",
        "\ntherefore, the reference answer",
        "\ntherefore, the correct answer is:",
        "\ntherefore, the answer is:",
        "\ntherefore, the final answer is:",
        "\nthe answer to the above question is:",
    ):
        i = tl.find(mk.lower())
        if i != -1:
            cut = min(cut, i)
    for cn in ("\n因此，正确答案是", "\n因此正确答案为", "\n正确答案是", "\n因此，答案为"):
        j = t.find(cn)
        if j != -1:
            cut = min(cut, j)
    if cut < len(t):
        t = t[:cut].rstrip()
    t = re.sub(r"\n\s*Answer\s*[:：]\s*[A-E]\b[^\n]*\s*$", "", t, flags=re.IGNORECASE)
    t = t.rstrip()
    if not t.strip() and raw.strip():
        return raw
    return t


def refine_assistant_content(
    raw: Optional[str],
    drop_prefix_chars: int = 0,
    keep_from_substring: str = "",
) -> str:
    """
    按配置去掉 role=assistant 的 content 前面部分（用于 CSV 与 prompt_mode=assistant / assistant_rag）。

    优先级：1) 若提供 keep_from_substring 且在文本中首次找到，则保留从该处起的后缀；
    2) 否则若 drop_prefix_chars>0，删除开头相应字符数；
    3) 否则原样返回。
    """
    if not raw or not str(raw).strip():
        return ""
    t = str(raw).strip()
    key = (keep_from_substring or "").strip()
    if key:
        i = t.find(key)
        if i != -1:
            return t[i:].strip()
    n = int(drop_prefix_chars or 0)
    if n > 0:
        if len(t) <= n:
            return ""
        return t[n:].lstrip()
    return t


def merge_system_and_user_body(system: Optional[str], body: str) -> str:
    """
    训练集 SFTDataset 仅含 user/assistant 轮次，无独立 system 槽位。
    将 system 与后续正文拼成单段，等价于在 user 里带系统提示的常见做法。
    """
    b = (body or "").strip()
    s = (system or "").strip()
    if s:
        return f"{s}\n\n{b}"
    return b


def format_inst_prompt_sft(user_block: str) -> str:
    """
    与 `dataset/lm_en_dataset.SFTDataset.build_prompt` 中 user 轮一致（空格位置一致）：
    `<s>[INST] {content} [/INST]` —— 模型从 [/INST] 后生成 assistant 续写。
    """
    content = user_block.strip()
    return f"<s>[INST] {content} [/INST]"


def format_llama2_inst_legacy(system: Optional[str], user: str) -> str:
    """旧版 Meta Llama2 对话：`<<SYS>>`；与本仓库 SFTDataset 字符串格式不一致，仅供对照实验。"""
    sys_block = (system or "").strip()
    usr = user.strip()
    if sys_block:
        return f"<s>[INST] <<SYS>>\n{sys_block}\n<</SYS>>\n\n{usr} [/INST]"
    return f"<s>[INST] {usr} [/INST]"


def _encode_len(tokenizer, text: str) -> int:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    return int(enc["input_ids"].shape[1])


def trim_segment_from_head_for_token_budget(
    tokenizer,
    max_input_len: int,
    prompt_template: str,
    system_txt: Optional[str],
    segment: str,
    prefix: str,
    suffix: str,
) -> str:
    """
    在 prefix / suffix **一字不改** 的前提下，仅从 segment **左侧**删去若干字符，
    使 compose(system, prefix+segment+suffix) 的 token 数 <= max_input_len。
    """
    segment = segment or ""
    prefix = prefix or ""
    suffix = suffix or ""

    def build_from_slice(mid_slice: str) -> str:
        body = prefix + mid_slice + suffix
        if prompt_template == "llama2_sys":
            return format_llama2_inst_legacy(system_txt, body)
        merged = merge_system_and_user_body(system_txt, body)
        return format_inst_prompt_sft(merged)

    if _encode_len(tokenizer, build_from_slice(segment)) <= max_input_len:
        return segment

    if _encode_len(tokenizer, build_from_slice("")) > max_input_len:
        print(
            "警告: 固定前缀+后缀（含说明/REFERENCE 边界/CHOICE_SUFFIX）已超过 max_input_len，"
            "无法仅靠删减参考正文适配；将依赖后续 tokenizer 截断。",
            file=sys.stderr,
        )
        return ""

    lo, hi = 0, len(segment)
    while lo < hi:
        mid = (lo + hi) // 2
        if _encode_len(tokenizer, build_from_slice(segment[mid:])) <= max_input_len:
            hi = mid
        else:
            lo = mid + 1
    # 从左侧多裁一点到下一行首，避免以半句/半段 Markdown 开头，利于小模型读 REFERENCE
    if 0 < lo < len(segment):
        nl = segment.find("\n", lo)
        if nl != -1 and nl + 1 < len(segment):
            lo2 = nl + 1
            if _encode_len(tokenizer, build_from_slice(segment[lo2:])) <= max_input_len:
                lo = lo2
    return segment[lo:]


def _encode_inst_body(
    tokenizer,
    prompt_template: str,
    system_txt: Optional[str],
    body: str,
) -> int:
    """与 build_prompt.compose 一致：整段 [INST] 内 body 的 token 数。"""
    if prompt_template == "llama2_sys":
        return _encode_len(tokenizer, format_llama2_inst_legacy(system_txt, body))
    merged = merge_system_and_user_body(system_txt, body)
    return _encode_len(tokenizer, format_inst_prompt_sft(merged))


def _max_inner_suffix_for_reference_budget(
    tokenizer,
    max_input_len: int,
    prompt_template: str,
    system_txt: Optional[str],
    q_block: str,
    ref_prefix: str,
    ref_suffix: str,
    inner_full: str,
) -> str:
    """
    只通过去掉 assistant 正文**前缀**来适配预算：返回 inner_full 的**最长后缀**，
    使 q_block + ref_prefix + suffix + ref_suffix 的整段 [INST] token 数 <= max_input_len。
    """
    inner_full = inner_full or ""
    if not inner_full:
        return ""

    def enc(ins: str) -> int:
        b = (q_block or "") + ref_prefix + ins + ref_suffix
        return _encode_inst_body(tokenizer, prompt_template, system_txt, b)

    if enc("") > max_input_len:
        return ""

    lo, hi = 0, len(inner_full)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        suf = inner_full[len(inner_full) - mid :] if mid else ""
        if enc(suf) <= max_input_len:
            best = suf
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_q_block_and_reference_inner(
    tokenizer,
    max_input_len: int,
    prompt_template: str,
    system_txt: Optional[str],
    q_block: str,
    ref_prefix: str,
    ref_suffix: str,
    inner_full: str,
) -> Tuple[str, str]:
    """
    assistant / assistant_rag：REFERENCE 内仅保留 assistant 的**后缀**（从左侧删），**绝不裁掉尾部**。
    若仅裁 assistant 仍无法进预算，再**从左侧**缩短 q_block；并保证 REFERENCE 内至少约 **MIN_REFERENCE_ASSISTANT_CHARS**
    个字符（避免只剩句末「.」等 1 个字符却被提前当作「已非空」返回）。
    """
    q0 = q_block or ""
    inner_full = inner_full or ""

    def pack(q: str, ins: str) -> int:
        return _encode_inst_body(
            tokenizer, prompt_template, system_txt, (q or "") + ref_prefix + ins + ref_suffix
        )

    if not inner_full.strip():
        return q0, ""

    if pack(q0, inner_full) <= max_input_len:
        return q0, inner_full

    target = min(MIN_REFERENCE_ASSISTANT_CHARS, len(inner_full.strip()))
    if target < 1:
        target = 1

    def suffix_for_q(lo: int) -> str:
        q_try = q0[lo:] if lo <= len(q0) else ""
        return _max_inner_suffix_for_reference_budget(
            tokenizer,
            max_input_len,
            prompt_template,
            system_txt,
            q_try,
            ref_prefix,
            ref_suffix,
            inner_full,
        )

    lo, hi = 0, len(q0) + 1
    while lo < hi:
        mid = (lo + hi) // 2
        suf = suffix_for_q(mid)
        if len(suf) >= target:
            hi = mid
        else:
            lo = mid + 1

    q_lo = min(lo, len(q0))
    q_out = q0[q_lo:]
    inner_out = suffix_for_q(q_lo)

    if len(inner_out) < target and inner_full.strip():
        q_out = ""
        inner_out = _max_inner_suffix_for_reference_budget(
            tokenizer,
            max_input_len,
            prompt_template,
            system_txt,
            "",
            ref_prefix,
            ref_suffix,
            inner_full,
        )

    if len(inner_out) < target and inner_full.strip():
        chunk = inner_full[-min(8192, len(inner_full)) :].lstrip()
        lo2, hi2 = 0, len(chunk) + 1
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            trial = chunk[mid:]
            if pack(q_out, trial) <= max_input_len:
                hi2 = mid
            else:
                lo2 = mid + 1
        inner_out = chunk[lo2:] if lo2 <= len(chunk) else chunk
        if len(inner_out) < min(32, len(inner_full.strip())):
            print(
                f"警告: REFERENCE 内 assistant 仅保留 {len(inner_out)} 字符（目标≥{target}）；"
                f"max_input_len={max_input_len} 过小或 system 过长，请增大上下文或减少 --max_new_tokens。",
                file=sys.stderr,
            )
    return q_out, inner_out


def trim_core_from_head_for_token_budget(
    tokenizer,
    max_input_len: int,
    prompt_template: str,
    system_txt: Optional[str],
    core_body: str,
    suffix: str,
) -> str:
    """
    在 **不改变 suffix 字符串一字** 的前提下，仅从 core_body **左侧**删去若干字符，
    使最终 prompt 的 token 数 <= max_input_len。

    core_body 对应 role=assistant（或将来扩展的「可变正文」）；suffix 含 CHOICE_SUFFIX_EN。
    """
    return trim_segment_from_head_for_token_budget(
        tokenizer,
        max_input_len,
        prompt_template,
        system_txt,
        core_body,
        "",
        suffix,
    )


def build_prompt(
    item: Dict[str, Any],
    prompt_mode: str,
    prompt_template: str = "sft",
    assistant_drop_prefix_chars: int = 0,
    assistant_keep_from_substring: str = "",
    tokenizer=None,
    max_input_len: Optional[int] = None,
    assistant_strip_conclusion: bool = False,
    assistant_rag_omit_stem: bool = False,
    assistant_rag_minimal_inst: bool = False,
    assistant_rag_slim_prompt: bool = False,
    assistant_rag_bare_prompt: bool = False,
) -> Tuple[str, Optional[str]]:
    suffix = _prompt_suffix_for_mode(prompt_mode)
    system_txt, user_txt, assistant_txt = parse_messages(item.get("messages", []))
    assistant_txt = refine_assistant_content(
        assistant_txt,
        assistant_drop_prefix_chars,
        assistant_keep_from_substring,
    )
    system_for_inst = system_txt
    if prompt_mode == "assistant_rag" and assistant_rag_minimal_inst:
        system_for_inst = None
    if prompt_mode == "assistant_rag" and assistant_rag_bare_prompt:
        # 与人设+题干+输出限制一致：bare 始终保留 system
        system_for_inst = system_txt

    def compose(body_with_suffix: str) -> str:
        if prompt_template == "llama2_sys":
            return format_llama2_inst_legacy(system_for_inst, body_with_suffix)
        merged = merge_system_and_user_body(system_for_inst, body_with_suffix)
        return format_inst_prompt_sft(merged)

    if prompt_mode == "rag":
        if not user_txt:
            raise ValueError("RAG 模式需要 messages 中含 role=user 的 content")
        user_body = user_txt
        if tokenizer is not None and max_input_len is not None:
            # 过长时从 **user 正文左侧** 丢弃，保留末尾与 CHOICE_SUFFIX、[/INST]（与 tokenizer truncation_side=left 一致，但先裁字符串更可控）
            user_body = trim_segment_from_head_for_token_budget(
                tokenizer,
                max_input_len,
                prompt_template,
                system_for_inst,
                user_txt,
                "",
                suffix,
            )
        return compose(user_body + suffix), None
    if prompt_mode == "assistant":
        if not assistant_txt:
            raise ValueError("assistant 模式需要 messages 中含 role=assistant 的 content")
        inner_full = assistant_txt
        if assistant_strip_conclusion:
            inner_full = strip_assistant_conclusion_for_extractor_prompt(inner_full)
        tail = _prompt_suffix_for_mode("assistant")
        ref_prefix = REFERENCE_ASSISTANT_BEGIN + "\n"
        ref_suffix = "\n" + REFERENCE_ASSISTANT_END + tail
        if tokenizer is not None and max_input_len is not None:
            q_block, inner_out = _fit_q_block_and_reference_inner(
                tokenizer,
                max_input_len,
                prompt_template,
                system_txt,
                MEDQA_EN_USER_INTRO_ASSISTANT_ONLY + MEDQA_EN_SECTION_ASSISTANT,
                ref_prefix,
                ref_suffix,
                inner_full,
            )
        else:
            q_block = MEDQA_EN_USER_INTRO_ASSISTANT_ONLY + MEDQA_EN_SECTION_ASSISTANT
            inner_out = inner_full
        inner_vis = inner_out.strip() if inner_out.strip() else inner_out
        if not inner_vis.strip() and assistant_txt.strip():
            inner_vis = assistant_txt[-min(4096, len(assistant_txt)) :].lstrip()
        ref_body = q_block + ref_prefix + inner_vis + ref_suffix
        return compose(ref_body), inner_vis
    if prompt_mode == "assistant_rag":
        if not assistant_txt:
            raise ValueError("assistant_rag 模式需要 messages 中含 role=assistant 的 content")
        q_body = (item.get("question_with_options") or "").strip()
        if not q_body:
            q_body = (user_txt or "").strip()
        if assistant_rag_bare_prompt:
            if not q_body:
                raise ValueError(
                    "assistant_rag + --assistant-rag-bare-prompt 需要 question_with_options 或 messages.user 题干"
                )
            # bare：不喂泄题尾段（The reference answer **C** / Therefore…）；CSV 对照仍用 JSON 全文（见下方 assistant_refined_full）
            inner_full = strip_assistant_conclusion_for_extractor_prompt(assistant_txt)
            tail = _prompt_suffix_for_mode("assistant_rag", bare=True)
            q_block = q_body + "\n\n"
            ref_prefix = ""
            ref_suffix = tail
            if tokenizer is not None and max_input_len is not None:
                q_block, inner_out = _fit_q_block_and_reference_inner(
                    tokenizer,
                    max_input_len,
                    prompt_template,
                    system_for_inst,
                    q_block,
                    ref_prefix,
                    ref_suffix,
                    inner_full,
                )
            else:
                inner_out = inner_full
            inner_vis = inner_out.strip() if inner_out.strip() else inner_out
            if not inner_vis.strip() and assistant_txt.strip():
                inner_vis = assistant_txt[-min(4096, len(assistant_txt)) :].lstrip()
            ref_body = q_block + inner_vis + ref_suffix
            return compose(ref_body), inner_vis

        omit_stem = (
            assistant_rag_omit_stem
            or assistant_rag_minimal_inst
            or assistant_rag_slim_prompt
        )
        if assistant_rag_minimal_inst:
            q_block = MEDQA_EN_USER_INTRO_ASSISTANT_RAG_NO_STEM + MEDQA_EN_SECTION_ASSISTANT
        elif assistant_rag_slim_prompt:
            # 仅 **Assistant analysis** + REFERENCE + 任务尾；system 仍经 merge 保留（与 --assistant-rag-minimal 不同）
            q_block = MEDQA_EN_SECTION_ASSISTANT
        elif assistant_rag_omit_stem:
            q_block = MEDQA_EN_USER_INTRO_ASSISTANT_RAG_NO_STEM + MEDQA_EN_SECTION_ASSISTANT
        elif q_body:
            q_block = (
                MEDQA_EN_USER_INTRO_ASSISTANT_RAG
                + MEDQA_EN_SECTION_QUESTION
                + q_body
                + "\n\n"
                + MEDQA_EN_SECTION_ASSISTANT
            )
        else:
            q_block = MEDQA_EN_USER_INTRO_ASSISTANT_RAG + MEDQA_EN_SECTION_ASSISTANT
        inner_full = assistant_txt
        if assistant_strip_conclusion:
            inner_full = strip_assistant_conclusion_for_extractor_prompt(inner_full)
        # slim：任务段用 CHOICE_SUFFIX_EN（与 CSV 中 [Task] 三行一致）；omit_stem/minimal 用 REF_ONLY；minimal+slim 同时开时以 minimal 为准仍用 REF_ONLY
        use_ref_only_task_suffix = omit_stem and (
            not assistant_rag_slim_prompt or assistant_rag_minimal_inst
        )
        tail = _prompt_suffix_for_mode(
            "assistant_rag", reference_only=use_ref_only_task_suffix
        )
        ref_prefix = REFERENCE_ASSISTANT_BEGIN + "\n"
        ref_suffix = "\n" + REFERENCE_ASSISTANT_END + tail
        if tokenizer is not None and max_input_len is not None:
            q_block, inner_out = _fit_q_block_and_reference_inner(
                tokenizer,
                max_input_len,
                prompt_template,
                system_for_inst,
                q_block,
                ref_prefix,
                ref_suffix,
                inner_full,
            )
        else:
            inner_out = inner_full
        inner_vis = inner_out.strip() if inner_out.strip() else inner_out
        if not inner_vis.strip() and assistant_txt.strip():
            inner_vis = assistant_txt[-min(4096, len(assistant_txt)) :].lstrip()
        ref_body = q_block + ref_prefix + inner_vis + ref_suffix
        return compose(ref_body), inner_vis
    if prompt_mode == "question_only":
        q = (item.get("question_with_options") or "").strip()
        if not q:
            raise ValueError("question_only 模式需要字段 question_with_options")
        q_body = q
        if tokenizer is not None and max_input_len is not None:
            q_body = trim_segment_from_head_for_token_budget(
                tokenizer,
                max_input_len,
                prompt_template,
                system_for_inst,
                q,
                "",
                suffix,
            )
        return compose(q_body + suffix), None
    raise ValueError(f"未知 prompt_mode: {prompt_mode}")


def probe_first_item_format(
    item: Dict[str, Any],
    prompt_mode: str,
    prompt_template: str,
    tokenizer,
    max_input_len: int,
    assistant_drop_prefix_chars: int = 0,
    assistant_keep_from_substring: str = "",
    assistant_strip_conclusion: bool = False,
    assistant_rag_omit_stem: bool = False,
    assistant_rag_minimal_inst: bool = False,
    assistant_rag_slim_prompt: bool = False,
    assistant_rag_bare_prompt: bool = False,
) -> None:
    """
    启动时打印首条样本的字段与 prompt 探针，便于确认「数据格式是否对齐」。
    若此处 build_prompt 成功且 prompt 以 [/INST] 结尾，则 JSON 形状通常无问题；
    若仍出现续写解码后为空（CSV 中 `model_raw_output` 为空串），多为预训练 Base 只预测换行/EOS，而非数据错。
    """
    keys = sorted(item.keys())
    raw_msgs = item.get("messages", [])
    if not isinstance(raw_msgs, list):
        print(
            f"【数据探针】警告: messages 不是 list（类型 {type(raw_msgs)}），将视为无消息。",
            file=sys.stderr,
        )
        raw_msgs = []
    roles: List[str] = []
    for m in raw_msgs:
        if isinstance(m, dict):
            roles.append(str(m.get("role", "")))
    st, ut, at = parse_messages(raw_msgs)
    at_ref = refine_assistant_content(
        at, assistant_drop_prefix_chars, assistant_keep_from_substring
    )
    print(
        f"【数据探针】首条 id={item.get('id')!r} 顶层键={keys}",
        file=sys.stderr,
    )
    print(
        f"【数据探针】messages 内 role 顺序={roles}；"
        f"解析得 lens: system={len(st or '')} user={len(ut or '')} "
        f"assistant_raw={len(at or '')} assistant_refined={len(at_ref)}",
        file=sys.stderr,
    )
    qw = item.get("question_with_options") or ""
    print(f"【数据探针】question_with_options 长度={len(str(qw))}", file=sys.stderr)
    try:
        prompt, asst_core_fit = build_prompt(
            item,
            prompt_mode,
            prompt_template,
            assistant_drop_prefix_chars,
            assistant_keep_from_substring,
            tokenizer,
            max_input_len,
            assistant_strip_conclusion=assistant_strip_conclusion,
            assistant_rag_omit_stem=assistant_rag_omit_stem,
            assistant_rag_minimal_inst=assistant_rag_minimal_inst,
            assistant_rag_slim_prompt=assistant_rag_slim_prompt,
            assistant_rag_bare_prompt=assistant_rag_bare_prompt,
        )
    except ValueError as e:
        print(f"【数据探针】build_prompt 失败（格式与 prompt_mode 不匹配）: {e}", file=sys.stderr)
        return
    if asst_core_fit is not None and len(asst_core_fit) < len(at_ref):
        print(
            f"【数据探针】assistant 参考正文因 max_input_len 从左侧裁掉约 {len(at_ref) - len(asst_core_fit)} 字符；"
            f"英文任务说明、REFERENCE 边界与 CHOICE_SUFFIX 未改。",
            file=sys.stderr,
        )
    ok_tail = prompt.rstrip().endswith("[/INST]")
    full_enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    full_tl = int(full_enc["input_ids"].shape[1])
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=False,
    )
    tl = int(enc["input_ids"].shape[1])
    trunc_dec = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)
    trunc_tail_ok = trunc_dec.rstrip().endswith("[/INST]")
    print(
        f"【数据探针】prompt 字符数={len(prompt)}；全文以 [/INST] 收尾={ok_tail}；"
        f"完整 token 数≈{full_tl}；截断后 tokens={tl}（max_input_len={max_input_len}）；"
        f"截断后文本仍以 [/INST] 收尾={trunc_tail_ok}",
        file=sys.stderr,
    )
    if full_tl > max_input_len and not trunc_tail_ok:
        print(
            "【数据探针】⚠ 截断后未保留末尾 [/INST]，请确认 tokenizer.truncation_side=left",
            file=sys.stderr,
        )
    if full_tl > max_input_len:
        drop = full_tl - max_input_len
        print(
            f"【数据探针】⚠ 输入超过 max_input_len：完整约 {full_tl} tokens，左侧截断丢弃约 {drop} tokens。"
            f"（max_input_len = max_seq_len - max_new_tokens - 8；可减小 --max_new_tokens 以留给 prompt 更多位置）",
            file=sys.stderr,
        )
    if not ok_tail:
        print(
            "【数据探针】警告: prompt 未以 [/INST] 结束，请检查模板或截断是否切断末尾。",
            file=sys.stderr,
        )


def extract_choice_from_model_output(text: str) -> str:
    """
    从 **model_raw_output**（模型续写）中解析 A–E；主流程下该列语义应为「从已分析正文中提取答案 + 末行 Answer」。
    使用较严规则，避免从正文中误抓孤立字母。
    """
    if not text or not text.strip():
        return ""
    t = text.strip()
    patterns = [
        r"(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer|最终答案)(?:\*\*)?\s*[:：]\s*\*?\*?\(?([A-E])\)?\.?",
        r"(?:Therefore|Thus|The answer is)[^\n.]{0,120}?[\s:\.\[\(]+([A-E])\b",
        r"(?:option|Option|选项)\s+([A-E])\b",
        r"Answer:\s*([A-E])\b",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines[-12:]):
        m = re.match(r"^[\*\s]*([A-E])\s*\.", ln)
        if m:
            return m.group(1).upper()
        m = re.match(r"^[\*\s]*(?:Answer|answer)\s*[:：]\s*([A-E])\b", ln, re.I)
        if m:
            return m.group(1).upper()
        m = re.match(r"^([A-E])$", ln)
        if m:
            return m.group(1).upper()
    return ""


def extract_choice_letter(text: str) -> str:
    """
    从任意英文说明/模型续写/assistant 正文中尽量提取 A–E 选项；失败时返回空字符串。
    策略：先匹配显式 Answer/最终答案 行，再扫最后几行中的单字母或 “X.” 行首，最后在尾部窗口找独立 A–E 词边界。
    """
    if not text or not text.strip():
        return ""
    t = text.strip()
    patterns = [
        r"(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer|最终答案)(?:\*\*)?\s*[:：]\s*\*?\*?\(?([A-E])\)?\.?",
        r"(?:Therefore|Thus|The answer is)[^\n.]{0,120}?[\s:\.\[\(]+([A-E])\b",
        r"(?:option|Option|选项)\s+([A-E])\b",
        r"Answer:\s*([A-E])\b",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines[-12:]):
        m = re.match(r"^[\*\s]*([A-E])\s*\.", ln)
        if m:
            return m.group(1).upper()
        m = re.match(r"^[\*\s]*(?:Answer|answer)\s*[:：]\s*([A-E])\b", ln, re.I)
        if m:
            return m.group(1).upper()
        m = re.match(r"^([A-E])$", ln)
        if m:
            return m.group(1).upper()

    tail = t[-800:] if len(t) > 800 else t
    ms = list(re.finditer(r"\b([A-E])\b", tail))
    if ms:
        return ms[-1].group(1).upper()
    return ""


def resolve_extracted_choice(
    assistant_content: str,
    model_raw_text: str,
    source: str,
) -> str:
    """
    主语义：`source=model` 时 **extracted_choice 只由 model_raw_text**（即 CSV 的 model_raw_output）解析；
    其它选项为从数据正文抄字母的对照路径。
    """
    a = extract_choice_letter((assistant_content or "").strip())
    m = extract_choice_from_model_output((model_raw_text or "").strip())
    if source == "model":
        return m
    if source == "assistant":
        return a
    if source == "both":
        return a if a else m
    return a


def _generation_ids_are_only_special_tokens(
    tokenizer, gen_ids: torch.Tensor
) -> bool:
    """续写段是否仅由 EOS/PAD/BOS 等特殊 id 构成（无可见正文）。"""
    if gen_ids.numel() == 0:
        return True
    spec: set[int] = set()
    for x in getattr(tokenizer, "all_special_ids", []) or []:
        spec.add(int(x))
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            spec.add(int(v))
    return all(int(t.item()) in spec for t in gen_ids.flatten())


def decode_generation_texts(
    tokenizer, gen_ids: torch.Tensor
) -> Tuple[str, str]:
    """
    返回 (用于从模型抽 A–E 的文本, 写入 CSV model_raw_output 的文本)。

    若第一步即 EOS 等，`skip_special_tokens=True` 解码常为空白；若整段仅有 `</s>` 等特殊符，
    两路皆返回空串，避免 CSV 里出现误导性的字面量 `</s>`（含义见 stderr [生成提示]）。
    """
    if gen_ids.numel() == 0:
        return "", ""
    if _generation_ids_are_only_special_tokens(tokenizer, gen_ids):
        return "", ""
    t_skip = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if t_skip.strip():
        return t_skip, t_skip
    t_all = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if t_all.strip():
        return t_skip, t_all.strip()
    ids = gen_ids.detach().cpu().tolist()
    return t_skip, f"[generated_token_ids={ids}]"


def postprocess_model_raw_output(text: str) -> str:
    """
    模型在 min_new_tokens 等约束下常反复吐换行 token，CSV 里会像「大量空字符」。
    压平过长空行；若已出现 `Answer: A–E` 收束行则裁到该行末尾，便于「短提取 + 结束」阅读。
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"\n{8,}", "\n\n\n", t)
    t = re.sub(r"\n{5,}", "\n\n", t)
    lines = t.split("\n")
    last_answer = -1
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if re.match(
            r"^(?:Answer|Final answer|最终答案)\s*[:：]\s*([A-E])\b", ln, re.I
        ):
            last_answer = i
            break
    if last_answer >= 0:
        t = "\n".join(lines[: last_answer + 1])
    t = t.strip()
    t = re.sub(r"\n{5,}", "\n\n", t)
    return t


def _stderr_hint_if_generation_blank(
    raw_text: str,
    gen_ids: torch.Tensor,
    eos_token_id: Optional[int],
    *,
    item_id: Any,
) -> None:
    """模型续写经 skip_special_tokens 解码为空时打 stderr；CSV 列由 decode_generation_texts 回退填充。"""
    if (raw_text or "").strip():
        return
    if gen_ids.numel() == 0:
        return
    preview = gen_ids[: min(12, gen_ids.numel())].tolist()
    extra = ""
    if (
        gen_ids.numel() == 1
        and eos_token_id is not None
        and int(gen_ids[0].item()) == int(eos_token_id)
    ):
        extra = "（首 token 为 EOS，常因对齐/模板导致立即结束）"
    elif preview and all(int(x) == 13 for x in preview[: min(8, len(preview))]):
        extra = "（多为换行 token 13）"
    print(
        f"[生成提示] id={item_id!r} 续写解码后为空或仅空白 n_tokens={int(gen_ids.numel())} "
        f"first_ids={preview} {extra}",
        file=sys.stderr,
    )


def _max_context(model) -> int:
    cfg = model.config
    for name in ("max_seq_len", "max_position_embeddings", "n_positions"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return 1024


def _load_causal_lm(model_path: str, device: torch.device, dtype: torch.dtype):
    """
    优先用 from_pretrained 加载。若当前 transformers 因 torch<2.6 拒绝加载 pytorch_model.bin，
    则回退为 from_config + 直接 torch.load 权重（仅用于本地信任权重；生产环境建议升级 torch>=2.6 或换 safetensors）。
    """
    attempts = (
        {"trust_remote_code": True, "dtype": dtype},
        {"trust_remote_code": True, "torch_dtype": dtype},
    )
    load_err: Optional[BaseException] = None
    for kw in attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
            # from_pretrained 默认留在 CPU；须与下方 enc.to(device) 一致
            return model.to(device=device, dtype=dtype)
        except TypeError:
            continue
        except ValueError as e:
            err_s = str(e)
            if "2.6" in err_s or "CVE-2025" in err_s or "torch.load" in err_s:
                load_err = e
                break
            raise

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise RuntimeError(
            "无法加载模型：from_pretrained 失败且未找到 pytorch_model.bin。\n"
            "请执行: pip install 'torch>=2.6'，或将权重转为 .safetensors 后再试。"
        ) from load_err

    print(
        "提示: 当前环境 torch<2.6，transformers 拒绝自动加载 .bin；"
        "已使用 from_config + torch.load 手动加载权重（适用于本地可信 checkpoint）。",
        file=sys.stderr,
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    try:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(bin_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"警告: load_state_dict 缺失键数量: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"警告: load_state_dict 多余键数量: {len(unexpected)}", file=sys.stderr)
    model = model.to(device=device, dtype=dtype)
    return model


def run():
    root = _project_root()
    parser = argparse.ArgumentParser(
        description=(
            "MedQA 等 JSON 批量推理：content 已含分析，将模型当作**答案提取器**；"
            "model_raw_output 存短续写（提取 + 末行 Answer），extracted_choice 默认仅从中解析 A–E。"
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(root / "parameters" / "QF_Struct_300M"),
        help=(
            "模型目录（含 config.json、pytorch_model.bin）。"
            "MedQA 指令评测请用 **QF_Struct_300M**（默认）；预训练 Base 不适合该模板。"
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(root / "dataset" / "RAG_MedQA_US_test_1273.json"),
        help="待推理的 MedQA RAG JSON 数组路径（默认 dataset/RAG_MedQA_US_test_1273.json，含 messages.user + messages.assistant）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(root / "inference_results.csv"),
        help="结果 CSV 路径",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        choices=("sft", "llama2_sys"),
        default="sft",
        help=(
            "sft: 与 trainer SFTDataset 一致——`<s>[INST] 单段正文 [/INST]`，system 合并进正文；"
            "tokenizer 使用 add_special_tokens=False。"
            "llama2_sys: 旧版 <<SYS>>（与仓库 SFT 训练字符串不一致）。"
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=("rag", "assistant", "assistant_rag", "question_only"),
        default="assistant_rag",
        help=(
            "rag: system+user 作 [INST] 输入；user 过长时从左侧裁切；"
            "assistant: 仅 REFERENCE 内包 assistant 分析，引导模型提取 A–E；"
            "assistant_rag（默认）: 先题干 question_with_options（无则 user），再 REFERENCE(assistant)，利于对齐选项；"
            "可用 --assistant-rag-omit-stem 省略 prompt 内题干块，仅 REFERENCE；"
            "可用 --assistant-rag-minimal 同时省略 system 与题干；"
            "可用 --assistant-rag-slim-prompt：保留 system，无题干与 MedQA 引导，仅 **Assistant analysis** + REFERENCE + 默认 [Task] 三行；"
            "可用 --assistant-rag-bare-prompt：保留 system + 题干 + 分析正文 + 简短 [Output]，无其余包裹；"
            "question_only: **不**使用 `messages` 里 role=assistant 的 content；[INST] 内仅 `question_with_options`（题干含 A–E）"
            "+ 下方 [Task] 输出约定；`system` 人设仍经合并保留。"
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help=(
            "最长续写 token 数。与「输入侧长度」抢预算："
            "max_input_len = max_seq_len - max_new_tokens - 8（见启动日志）。"
            "RAG 长文可把本值降到 128–256 以显著增加可放进 prompt 的 token 数。"
        ),
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=12,
        help=(
            "传给 model.generate 的 min_new_tokens（默认 12）。"
            "过小或为 0 时常出现续写**第一个 token 即 EOS**，extracted_choice 为空；"
            "过大时小模型可能刷换行 token，CSV 会显空行多，可配合 postprocess 与 --no-compact-model-output。"
            "若需旧行为可显式传 --min_new_tokens 0。"
        ),
    )
    parser.add_argument(
        "--extract_choice_from",
        type=str,
        choices=("assistant", "model", "both"),
        default="model",
        help=(
            "extracted_choice：model=**仅**从 model_raw_output 解析 A–E（主语义，与列名一致）；"
            "assistant=从 JSON assistant 正文解析（对照用）；both=先 assistant 再 model。"
        ),
    )
    parser.add_argument(
        "--no-compact-model-output",
        action="store_true",
        help=(
            "默认会对 model_raw_output 做压缩：压平过长空行、若已有 Answer: A–E 行则裁到该行末尾。"
            "加本开关则写入 decode 原文（可能含大量换行空字符）。"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="只跑前 N 条，0 表示全部"
    )
    parser.add_argument(
        "--skip-data-probe",
        action="store_true",
        help="跳过启动时对首条样本的数据/格式探针（默认会打印到 stderr 便于对齐检查）",
    )
    parser.add_argument(
        "--no-use-cache",
        dest="no_use_cache",
        action="store_true",
        help="禁用 KV cache（解码会很慢；默认启用 KV cache）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda / cpu，空则自动",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="GPU 上使用 float16（更快；当前 qbmodel 仅对 Attention 做 bf16 autocast，易与 FFN 出现 float/half 混用报错，默认关闭）",
    )
    parser.add_argument(
        "--assistant_keep_from_substring",
        type=str,
        default="",
        help=(
            "对 messages 中 role=assistant 的 content：从该子串在文中**首次出现**处起保留（含子串），去掉其前面的部分；"
            "空字符串表示不启用。MedQA 常配合 **Analysis:** 以去掉 'In response to your question...' 前摇。"
            "若未找到子串，则回退为 --assistant_drop_prefix_chars（若为 0 则仍用全文）。"
        ),
    )
    parser.add_argument(
        "--assistant_drop_prefix_chars",
        type=int,
        default=0,
        help=(
            "去掉 assistant content 开头若干字符；在 keep_from_substring 未匹配或为空时生效。"
            "0 表示不按长度截断。"
        ),
    )
    parser.add_argument(
        "--assistant-strip-conclusion",
        action="store_true",
        help=(
            "在 assistant / assistant_rag 下从 REFERENCE（及 bare 布局中的 assistant 段）内裁掉末尾泄题/收束段："
            "The reference answer…、Therefore/因此…、末行 Answer: A–E 等（减轻 prompt 已写死选项仍「提取」或首 token EOS）。"
            "默认关闭；**`--assistant-rag-bare-prompt` 会始终做此类裁剪**（无需再开本开关）。"
        ),
    )
    parser.add_argument(
        "--assistant-rag-omit-stem",
        action="store_true",
        help=(
            "仅当 --prompt_mode assistant_rag 时生效：不在 [INST] 内重复 **Stem and options** 与 question_with_options/user 题干，"
            "只保留简短说明 + **Assistant analysis** + REFERENCE（减轻与选项列表重复、省 token）。"
            "题干仍写入 CSV 的 question_with_options 列。"
        ),
    )
    parser.add_argument(
        "--assistant-rag-minimal",
        action="store_true",
        help=(
            "仅当 --prompt_mode assistant_rag 时生效：[INST] 内**不**拼接 messages 里的 system 人设长段，也**不**重复题干"
            "（等价于同时省掉 system + **Stem and options**；与单独开 --assistant-rag-omit-stem 相比再多省 system）。"
            "仅保留英文短引导 + **Assistant analysis** + REFERENCE + 任务尾，便于把 token 预算留给 REFERENCE 正文。"
            "题干与 system 仍可从 CSV 的 question_with_options / 数据 JSON 查看。"
        ),
    )
    parser.add_argument(
        "--assistant-rag-slim-prompt",
        action="store_true",
        help=(
            "仅当 --prompt_mode assistant_rag 时生效：保留 messages 里 **system**（如 You are a professional doctor…），"
            "但不拼题干；[INST] 内 user 正文仅 **Assistant analysis** + REFERENCE + 末尾 [Task] 三行（与默认 CHOICE_SUFFIX_EN 一致，"
            "含 “analysis + stem” 表述）。不含 “MedQA sample…” 英文引导段。"
            "与 --assistant-rag-omit-stem 同时开启时以本开关为准（更短前缀）。与 --assistant-rag-minimal 同时开启时以 minimal 为准。"
        ),
    )
    parser.add_argument(
        "--assistant-rag-bare-prompt",
        action="store_true",
        help=(
            "仅当 --prompt_mode assistant_rag 时生效：[INST] 内**仅** system（人设，经合并）+ 题干原文 + assistant 正文 + 简短 [Output] 格式说明；"
            "无 MedQA 引导、无 **Stem/Assistant** 标题、无 <<<REFERENCE>>> 标记、无 “(Write only after [/INST]…)”。"
            "须具备 question_with_options 或 messages.user 题干。"
            "会对 assistant 正文去掉 **The reference answer…** / **Therefore…** 等泄题尾段后再喂模型（不删 JSON 原数据）。"
        ),
    )
    parser.add_argument(
        "--no-csv-model-input",
        action="store_true",
        help="不在 CSV 中写入 model_input_prompt 列（该列填空），以减小文件体积。默认会写入截断后实际喂给模型的 prompt。",
    )
    args = parser.parse_args()

    if args.assistant_rag_omit_stem and args.prompt_mode != "assistant_rag":
        print(
            "提示: --assistant-rag-omit-stem 仅作用于 --prompt_mode assistant_rag，当前模式下已忽略。",
            file=sys.stderr,
        )
    if args.assistant_rag_minimal and args.prompt_mode != "assistant_rag":
        print(
            "提示: --assistant-rag-minimal 仅作用于 --prompt_mode assistant_rag，当前模式下已忽略。",
            file=sys.stderr,
        )
    if args.assistant_rag_slim_prompt and args.prompt_mode != "assistant_rag":
        print(
            "提示: --assistant-rag-slim-prompt 仅作用于 --prompt_mode assistant_rag，当前模式下已忽略。",
            file=sys.stderr,
        )
    if args.assistant_rag_bare_prompt and args.prompt_mode != "assistant_rag":
        print(
            "提示: --assistant-rag-bare-prompt 仅作用于 --prompt_mode assistant_rag，当前模式下已忽略。",
            file=sys.stderr,
        )
    if args.assistant_rag_minimal and args.assistant_rag_slim_prompt:
        print(
            "提示: 同时指定 --assistant-rag-minimal 与 --assistant-rag-slim-prompt 时，layout 以 minimal 为准（无 system）。",
            file=sys.stderr,
        )
    if args.assistant_rag_bare_prompt and args.assistant_rag_minimal:
        print(
            "提示: 已同时指定 --assistant-rag-bare-prompt 与 --assistant-rag-minimal：bare 会保留 system 人设与题干，minimal 的「无人设不生效」。",
            file=sys.stderr,
        )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        print(f"错误：模型目录不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    base_name = os.path.basename(os.path.normpath(model_path))
    if "_Base_" in base_name:
        print(
            "警告: 当前指向预训练 Base 权重。对 `[INST]` 选择题推理通常应使用 **QF_Struct_300M**（指令微调），"
            "否则易出现仅换行等退化输出，看起来像 model_raw_output 为空（实为 strip 后无可见字符）。",
            file=sys.stderr,
        )

    print(f"加载分词器与模型: {model_path} ；设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # 超长时必须保留 prompt **末尾**（含 [/INST]）；HF 默认 truncation_side=right 会从末尾删 token，会裁掉 [/INST]
    tokenizer.truncation_side = "left"

    # 默认 float32：与 qbmodel 中局部 autocast 兼容，避免 mat1 float / mat2 half
    dtype = (
        torch.float16 if (device.type == "cuda" and args.fp16) else torch.float32
    )
    print(f"模型计算 dtype: {dtype}", flush=True)
    if args.prompt_mode in ("assistant", "assistant_rag"):
        print(
            "REFERENCE 内 assistant 末尾收束段（Therefore/因此…）裁剪: "
            f"{'开启' if args.assistant_strip_conclusion else '关闭（默认，保留 Therefore… 等尾段）'}",
            flush=True,
        )
    if args.prompt_mode == "assistant_rag" and args.assistant_rag_bare_prompt:
        print(
            "assistant_rag: 已开启 --assistant-rag-bare-prompt（仅人设 + 题干 + 分析正文 + [Output]；无 REFERENCE/MedQA 等包裹）。",
            flush=True,
        )
    elif args.prompt_mode == "assistant_rag" and args.assistant_rag_minimal:
        print(
            "assistant_rag: 已开启 --assistant-rag-minimal（[INST] 内无 system、无题干，仅分析+REFERENCE+任务尾）。",
            flush=True,
        )
    elif args.prompt_mode == "assistant_rag" and args.assistant_rag_slim_prompt:
        print(
            "assistant_rag: 已开启 --assistant-rag-slim-prompt（保留 system；无题干与 MedQA 引导，仅 **Assistant analysis** + REFERENCE + [Task] 三行）。",
            flush=True,
        )
    elif args.prompt_mode == "assistant_rag" and args.assistant_rag_omit_stem:
        print(
            "assistant_rag: 已开启 --assistant-rag-omit-stem（prompt 内不重复题干，仍带 system）。",
            flush=True,
        )
    if args.prompt_mode == "question_only":
        print(
            "question_only: 仅人设 + 题干 + A–E 输出说明；不读取 messages 中 assistant 分析正文。",
            flush=True,
        )
    print(
        f"prompt 模板: {args.prompt_template}（sft 对齐 dataset/lm_en_dataset.SFTDataset）",
        flush=True,
    )
    use_kv = not args.no_use_cache
    print(
        f"KV cache（use_cache）: {'开启 — 推荐（解码快）' if use_kv else '关闭 — 仅调试时建议'}",
        flush=True,
    )
    print(
        f"extracted_choice 来源: {args.extract_choice_from} "
        f"（model=仅从 model_raw_output 解析 A–E，主语义；assistant/both=从数据正文对照）",
        flush=True,
    )
    if args.extract_choice_from == "model":
        print(
            "列语义: model_raw_output=从已分析 content 中**提取**答案的短续写（+ 末行 Answer: ）；"
            "extracted_choice=仅从中抽取 A–E。",
            flush=True,
        )
    else:
        print(
            "列语义: model_raw_output 仍为模型续写；extracted_choice 当前从 assistant/both 解析（对照路径）。",
            flush=True,
        )
    if args.no_csv_model_input:
        print(
            "CSV 列 model_input_prompt: 已禁用（--no-csv-model-input），该列将留空。",
            flush=True,
        )
    else:
        print(
            "CSV 列 model_input_prompt: 写入经 tokenizer 截断后实际参与前向的完整 prompt（由 input_ids 解码）。",
            flush=True,
        )
    model = _load_causal_lm(model_path, device, dtype)
    model.eval()

    if device.type == "cpu":
        for m in model.modules():
            if m.__class__.__name__ == "QubitAttention":
                m.flash = False

    data_path = os.path.abspath(args.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("错误：JSON 根应为数组", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        data = data[: args.limit]

    def _item_has_refined_assistant(it: Dict[str, Any]) -> bool:
        _, _, raw_a = parse_messages(it.get("messages", []))
        r = refine_assistant_content(
            raw_a,
            args.assistant_drop_prefix_chars,
            args.assistant_keep_from_substring,
        )
        return bool(r.strip())

    if data and args.prompt_mode in ("assistant", "assistant_rag"):
        if not _item_has_refined_assistant(data[0]):
            print(
                "错误: 当前 --prompt_mode 为 assistant/assistant_rag，但首条样本的 messages 中"
                "无 role=assistant 或 content 经裁剪后为空。请换带 assistant 的 JSON，或改用 --prompt_mode rag。",
                file=sys.stderr,
            )
            sys.exit(1)
    elif data and args.prompt_mode == "rag" and not _item_has_refined_assistant(data[0]):
        print(
            "提示: 首条样本无可用 role=assistant。若要做「从已写好的分析里提取 A–E」，"
            "请使用含 assistant 的 MedQA JSON；默认 --prompt_mode assistant_rag 即为此场景。",
            file=sys.stderr,
        )

    max_ctx = _max_context(model)
    max_input_len = max(64, max_ctx - args.max_new_tokens - 8)
    print(
        f"上下文预算: max_seq_len≈{max_ctx}；max_input_len={max_input_len} "
        f"（= max_seq_len - max_new_tokens({args.max_new_tokens}) - 8）",
        flush=True,
    )
    if args.min_new_tokens <= 0:
        print(
            "提示: --min_new_tokens=0 时易出现续写首 token 即 EOS，model_raw_output 可能为空；"
            "若从 assistant 提取答案建议保持默认 12 或更大。",
            flush=True,
        )

    if not args.skip_data_probe and data:
        probe_first_item_format(
            data[0],
            args.prompt_mode,
            args.prompt_template,
            tokenizer,
            max_input_len,
            args.assistant_drop_prefix_chars,
            args.assistant_keep_from_substring,
            assistant_strip_conclusion=args.assistant_strip_conclusion,
            assistant_rag_omit_stem=args.assistant_rag_omit_stem,
            assistant_rag_minimal_inst=args.assistant_rag_minimal,
            assistant_rag_slim_prompt=args.assistant_rag_slim_prompt,
            assistant_rag_bare_prompt=args.assistant_rag_bare_prompt,
        )

    out_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        # model_raw_output: 短续写（从已分析 content 提取答案 + 末行 Answer:）
        # extracted_choice: 默认仅从 model_raw_output 解析的 A–E
        "id",
        "meta_info",
        "question_with_options",
        "assistant_content",
        "answer_with_idx",
        "answer_idx",
        "model_input_prompt",
        "model_raw_output",
        "extracted_choice",
        "matches_answer_idx",
        "error",
    ]

    n_written = 0
    blank_generation_count = 0
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        fout.flush()

        for item in tqdm(data, desc="推理"):
            qwo = item.get("question_with_options", "")
            awi = item.get("answer_with_idx", "")
            aidx = (item.get("answer_idx") or "").strip().upper()
            rec_id = item.get("id", "")
            meta = item.get("meta_info", "")
            _, _, raw_asst = parse_messages(item.get("messages", []))
            assistant_refined_full = refine_assistant_content(
                raw_asst,
                args.assistant_drop_prefix_chars,
                args.assistant_keep_from_substring,
            )
            assistant_content = assistant_refined_full

            try:
                prompt, asst_core_fit = build_prompt(
                    item,
                    args.prompt_mode,
                    args.prompt_template,
                    args.assistant_drop_prefix_chars,
                    args.assistant_keep_from_substring,
                    tokenizer,
                    max_input_len,
                    assistant_strip_conclusion=args.assistant_strip_conclusion,
                    assistant_rag_omit_stem=args.assistant_rag_omit_stem,
                    assistant_rag_minimal_inst=args.assistant_rag_minimal,
                    assistant_rag_slim_prompt=args.assistant_rag_slim_prompt,
                    assistant_rag_bare_prompt=args.assistant_rag_bare_prompt,
                )
                if asst_core_fit is not None:
                    assistant_content = asst_core_fit
            except ValueError as e:
                writer.writerow(
                    {
                        "id": rec_id,
                        "meta_info": meta,
                        "question_with_options": qwo,
                        "assistant_content": assistant_content,
                        "answer_with_idx": awi,
                        "answer_idx": aidx,
                        "model_input_prompt": "",
                        "model_raw_output": "",
                        "extracted_choice": "",
                        "matches_answer_idx": False,
                        "error": str(e),
                    }
                )
                fout.flush()
                n_written += 1
                continue

            # 与 SFTDataset 一致：字符串内已含 `<s>`，勿再自动加 BOS（否则会 `<s><s>[INST]...`）
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            if args.no_csv_model_input:
                model_input_prompt = ""
            else:
                model_input_prompt = tokenizer.decode(
                    enc["input_ids"][0],
                    skip_special_tokens=False,
                )

            with torch.no_grad():
                gen_kw = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=not args.no_use_cache,
                )
                if args.min_new_tokens and args.min_new_tokens > 0:
                    gen_kw["min_new_tokens"] = min(
                        args.min_new_tokens, args.max_new_tokens
                    )
                out = model.generate(**enc, **gen_kw)
            gen_ids = out[0, in_len:]
            raw_extract, raw_csv = decode_generation_texts(tokenizer, gen_ids)
            canonical = raw_csv if not (raw_extract or "").strip() else raw_extract
            out_text = (
                postprocess_model_raw_output(canonical)
                if not args.no_compact_model_output
                else canonical
            )
            if not (raw_extract or "").strip() and gen_ids.numel() > 0:
                blank_generation_count += 1
                if blank_generation_count <= 3:
                    _stderr_hint_if_generation_blank(
                        raw_extract,
                        gen_ids,
                        tokenizer.eos_token_id,
                        item_id=rec_id,
                    )
            ex = resolve_extracted_choice(
                assistant_refined_full, out_text, args.extract_choice_from
            )
            match = bool(ex) and (ex == aidx)

            writer.writerow(
                {
                    "id": rec_id,
                    "meta_info": meta,
                    "question_with_options": qwo,
                    "assistant_content": assistant_content,
                    "answer_with_idx": awi,
                    "answer_idx": aidx,
                    "model_input_prompt": model_input_prompt,
                    "model_raw_output": out_text,
                    "extracted_choice": ex,
                    "matches_answer_idx": match,
                    "error": "",
                }
            )
            fout.flush()
            n_written += 1

        if blank_generation_count:
            print(
                f"[生成提示] 共 {blank_generation_count} 条样本续写 decode 后为空或仅空白；"
                "CSV 中 model_raw_output 经压缩后仍可能为空。可试 --min_new_tokens、--prompt_mode assistant_rag 或 assistant。",
                file=sys.stderr,
            )

    print(f"已保存: {out_path} ，共 {n_written} 行")


if __name__ == "__main__":
    run()
