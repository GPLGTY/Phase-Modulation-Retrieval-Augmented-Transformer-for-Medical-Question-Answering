#!/usr/bin/env python3
"""From retrieval JSON → Intent samples (one LLM call).

Supports cloud or local OpenAI-compatible endpoints. RAG is optional.
Resume from {out_json}.partial.json by sample id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Fill these in. CLI flags override the same fields.
# Local vLLM/Ollama/LM Studio: set BASE_URL to the OpenAI-compatible root
# (often .../v1) and API_KEY to any non-empty placeholder such as "EMPTY".
# ---------------------------------------------------------------------------
CONFIG = {
    "base_url": os.getenv(
        "VOLCENGINE_ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
    ),
    # 勿在仓库中写入真实密钥；请用环境变量或 --api_key
    "api_key": os.getenv("VOLCENGINE_ARK_API_KEY") or os.getenv("ARK_API_KEY") or "",
    "model": os.getenv("INTENT_REWRITE_MODEL", "doubao-seed-2-0-mini-260428"),
    "workers": 8,
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 3600.0,
}

SYSTEM = (
    "You are an expert in intent recognition. Based on the provided information, "
    "you can determine which of the options A through E the intent of the given "
    "text corresponds to."
)
LEAD = "In response to your question, my analysis process and answer are as follows:"

PROMPT_NO_RAG = """You write the user message for an intent-recognition example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Analyze the question and options using your own medical knowledge, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.

Question:
{question_with_options}
"""

PROMPT_RAG = """You write the user message for an intent-recognition example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Below are medical knowledge passages retrieved from reference textbooks; they may or may not be relevant — use them only when helpful. Analyze the question and options, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question. Do not paste or quote long retrieved passages; paraphrase only what you use.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.
Do NOT include a "#Medical knowledge#" section or any raw retrieved text in your output.

Retrieved knowledge (for your reasoning only; do not copy into the output):
<knowledge>
{knowledges_for_llm}
</knowledge>

Question:
{question_with_options}
"""

INTENT_RE = re.compile(
    r"(?:Therefore[, ]+(?:the )?(?:correct )?answer(?: to the (?:above )?question(?: above)?)? is\s*[:：]?\s*"
    r"(?:\*\*)?([A-E])(?:\*\*)?"
    r"|the (?:correct )?answer is\s*[:：]?\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|correct choice is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|best (?:choice|option|answer) is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|chosen option is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|Answer\s*[:：]\s*(?:\*\*)?([A-E])(?:\*\*)?)",
    re.IGNORECASE,
)
LETTER_RE = re.compile(r"\b([A-E])\b")
OPTION_RE = re.compile(r"(?m)^\s*([A-E])[\.\)]\s*(.+?)\s*$")
BARE_THEREFORE = "Therefore, the answer to the above question is: "
THEREFORE_LINE_RE = re.compile(
    r"(?:Therefore,\s*)?(?:the )?answer to (?:the )?(?:above )?question is\s*[:：].*$",
    re.IGNORECASE | re.DOTALL,
)


def knowledges_for_llm(sample: dict) -> str:
    if sample.get("knowledges_for_llm"):
        return str(sample["knowledges_for_llm"]).strip()
    ri = sample.get("retrieve_info") or {}
    ki = ri.get("knowledges_info") or {}
    knowledges = ki.get("knowledges") if isinstance(ki, dict) else None
    if isinstance(knowledges, list) and knowledges:
        return "\n\n".join(str(x) for x in knowledges if x)
    return ""


def rag_k(sample: dict, with_rag: bool) -> int:
    if not with_rag:
        return 0
    ri = sample.get("retrieve_info") or {}
    declared = ri.get("topk_knowledge")
    if isinstance(declared, int) and declared > 0:
        return declared
    ki = (ri.get("knowledges_info") or {}).get("knowledges") or []
    return len(ki) if isinstance(ki, list) else 0


def gold_letter(sample: dict) -> str:
    g = str(sample.get("gold_letter") or sample.get("answer_idx") or "").strip().upper()
    if g in "ABCDE":
        return g
    return g[0] if g and g[0] in "ABCDE" else ""


def _norm_phrase(text: str) -> str:
    text = (text or "").lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_options(question: str) -> dict[str, str]:
    return {
        m.group(1).upper(): m.group(2).strip()
        for m in OPTION_RE.finditer(question or "")
    }


def _conclusion(text: str) -> str:
    m = re.search(r"\*\*Conclusion:\*\*([\s\S]*)$", text or "", re.I)
    if not m:
        m = re.search(r"(?m)^Conclusion:([\s\S]*)$", text or "", re.I)
    return m.group(1) if m else ""


def _significant_tokens(text: str) -> set[str]:
    return {tok for tok in _norm_phrase(text).split() if len(tok) > 3}


def _letter_conflicts(text: str, letter: str, options: dict[str, str]) -> bool:
    """A bold 'A. some other option' is not the question's option A."""
    real = options.get(letter) or ""
    if not real:
        return False
    stated = re.findall(rf"\*\*\s*{letter}\.\s*([^*\n]+)", text or "", flags=re.I)
    if not stated:
        return False
    real_toks = _significant_tokens(real)
    if not real_toks:
        return False
    for chunk in stated:
        got = _significant_tokens(chunk)
        if got and got.isdisjoint(real_toks):
            return True
    return False


def _letter_from_option_names(text: str, question: str) -> str | None:
    """Conclusion names the option text ('Chi square test') but not its letter."""
    options = parse_options(question)
    conc = _conclusion(text)
    if not options or not conc:
        return None
    conc = THEREFORE_LINE_RE.sub(" ", conc)
    sentences = re.split(r"(?<=[.!?])\s+", conc)
    choice = [
        s
        for s in sentences
        if re.search(r"best|correct|appropriate|answer|choice", s, re.I)
    ]
    regions = [_norm_phrase(s) for s in choice] or [_norm_phrase(conc)]
    hits: list[tuple[int, str]] = []
    for letter, name in options.items():
        norm_name = _norm_phrase(name)
        if len(norm_name) < 4:
            continue
        if any(norm_name in region for region in regions):
            hits.append((len(norm_name), letter))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


def extract_intent_letter(text: str, question: str = "") -> str | None:
    hits = []
    for m in INTENT_RE.finditer(text or ""):
        g = next((x for x in m.groups() if x), None)
        if g:
            hits.append(g.upper())
    letter = hits[-1] if hits else None
    if not letter:
        conc = _conclusion(text or "")
        tail = conc or (text or "")[-500:]
        m2 = re.findall(r"\*\*\s*([A-E])\.\s+", tail)
        if m2:
            letter = m2[-1].upper()
        else:
            letters = LETTER_RE.findall(tail)
            letter = letters[-1].upper() if letters else None
    options = parse_options(question)
    if letter and _letter_conflicts(text or "", letter, options):
        letter = None
    if letter:
        return letter
    return _letter_from_option_names(text or "", question)


def _is_repetition(text: str) -> bool:
    raw = (text or "").strip()
    if len(raw) < 400:
        return False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 8:
        keys = [re.sub(r"\s+", " ", ln.lower())[:120] for ln in lines]
        top = max(keys.count(k) for k in set(keys))
        if top >= 5 and top / len(keys) >= 0.35:
            return True
    parts = [p.strip() for p in re.split(r"[\n.]+", raw) if len(p.strip()) > 40]
    if len(parts) >= 8:
        top = max(parts.count(p) for p in set(parts))
        if top >= 5 and top / len(parts) >= 0.4:
            return True
    for size in (32, 64):
        if len(raw) < size * 10:
            continue
        counts: dict[str, int] = {}
        for i in range(0, min(len(raw) - size, 12000), size):
            piece = raw[i : i + size]
            counts[piece] = counts.get(piece, 0) + 1
        top = max(counts.values(), default=0)
        if top >= 8 and top * size / len(raw) >= 0.25:
            return True
    return False


def _force_bare_therefore(text: str) -> str:
    matches = list(THEREFORE_LINE_RE.finditer(text))
    if matches:
        m = matches[-1]
        tail = text[m.start() :]
        if len(tail) < 500 or m.start() > len(text) * 0.45:
            text = text[: m.start()].rstrip() + "\n\n" + BARE_THEREFORE
    if not text.endswith(BARE_THEREFORE) and re.search(r"Conclusion:", text, re.I):
        text = text.rstrip() + "\n\n" + BARE_THEREFORE
    if text.endswith(BARE_THEREFORE):
        return text.lstrip()
    return text.strip()


def clean_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"</sam>\s*", "", text, flags=re.I)
    text = re.sub(r"</?think>\s*", "", text, flags=re.I)
    text = re.sub(r"^(?:User|USER|user message)\s*:\s*", "", text, flags=re.I)
    if not text.startswith(LEAD):
        body = text.lstrip()
        if body.startswith("**Analysis:**") or body.startswith("Analysis"):
            text = LEAD + "\n\n" + body
    text = re.sub(
        r"(?is)#\s*Medical knowledge\s*#.*?(?=\n\s*#\s*Medical question\s*#|\n\s*\*\*Analysis:\*\*|\n\s*Analysis:)",
        "",
        text,
    )
    text = re.sub(r"(?is)<knowledge>.*?</knowledge>\s*", "", text)
    return _force_bare_therefore(text)


def make_prompt(sample: dict, with_rag: bool) -> str:
    q = sample.get("question_with_options") or ""
    if with_rag:
        kb = knowledges_for_llm(sample)
        if not kb:
            raise RuntimeError("with_rag set but knowledges empty")
        return PROMPT_RAG.format(knowledges_for_llm=kb, question_with_options=q)
    return PROMPT_NO_RAG.format(question_with_options=q)


def _message_text(msg) -> str:
    """Prefer the formal answer. If it is empty, use the reasoning trace."""
    content = (getattr(msg, "content", None) or "").strip()
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or ""
    reasoning = str(reasoning or "").strip()
    if _is_repetition(content):
        content = ""
    if _is_repetition(reasoning):
        reasoning = ""
    if content and "Analysis" in content and "Conclusion" in content:
        return content
    if reasoning and "Analysis" in reasoning and "Conclusion" in reasoning:
        return reasoning
    if content:
        return content
    return reasoning


def repair_prompt(question: str, notes: str) -> str:
    options = parse_options(question)
    option_lines = "\n".join(f"{letter}. {name}" for letter, name in options.items())
    clipped = (notes or "").strip()[:4000]
    return f"""Rewrite the scratch notes into the required user message. Output only that message, in English. No scratch notes, no title, no JSON.

Start with this exact text:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

Then write 3 to 5 numbered points and one bullet for every option below, copying the letter and option text exactly. Do not add or rename options.

**Conclusion:**
In 2 to 4 sentences, name the chosen option with its letter and its text.

Therefore, the answer to the above question is: 

The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon and nothing else on that line. Do not repeat a sentence.

Options:
{option_lines}

Question:
{question}

Scratch notes to rewrite, not to paste:
{clipped}
"""


def option_lock(question: str) -> str:
    options = parse_options(question)
    if not options:
        return ""
    lines = "\n".join(f"{letter}. {name}" for letter, name in options.items())
    return (
        "\n\nHard constraint: copy the option bullets verbatim from this list. "
        "Do not add, remove, rename, or replace any option. "
        "Start the output with **Analysis:**. Do not repeat any sentence or line.\n"
        + lines
    )


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if presence_penalty:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty:
        kwargs["frequency_penalty"] = frequency_penalty
    if repetition_penalty and repetition_penalty != 1.0:
        kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}
    r = client.chat.completions.create(**kwargs)
    return clean_output(_message_text(r.choices[0].message))


def build_sample(
    src: dict,
    user_text: str,
    model: str,
    with_rag: bool,
    source: str,
) -> dict:
    intent = extract_intent_letter(user_text, src.get("question_with_options") or "")
    if not intent:
        raise RuntimeError("unresolved intent_letter")
    gold = gold_letter(src)
    return {
        "id": src.get("id"),
        "meta_info": src.get("meta_info") or src.get("subject_name"),
        "question_with_options": src.get("question_with_options"),
        "answer_with_idx": src.get("answer_with_idx"),
        "answer_idx": gold or intent,
        "gold_letter": gold or None,
        "intent_letter": intent,
        "intent_extract_source": "rewrite_conclusion",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": intent},
        ],
        "model_prompt_text": f"[INST] {SYSTEM}\n\n{user_text} [/INST]",
        "source": source,
        "user_rewrite_model": model,
        "user_rewrite_no_gold": True,
        "user_rewrite_with_rag": bool(with_rag),
    }


def load_done(partial_path: Path) -> dict[str, dict]:
    if not partial_path.exists():
        return {}
    raw = json.loads(partial_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "samples" in raw:
        items = raw["samples"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    done = {}
    for s in items:
        sid = str(s.get("id"))
        if sid and sid != "None":
            done[sid] = s
    return done


def save_partial(partial_path: Path, done: dict[str, dict], meta: dict) -> None:
    payload = {"meta": meta, "samples": [done[k] for k in sorted(done, key=str)]}
    partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval JSON → intent JSON (OpenAI-compatible API)."
    )
    p.add_argument("--in_json", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--with_rag", action="store_true", help="Put retrieved passages in the LLM prompt only")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=0, help="0 = remaining samples after offset")
    p.add_argument("--model", default=CONFIG["model"])
    p.add_argument("--base_url", default=CONFIG["base_url"])
    p.add_argument("--api_key", default=CONFIG["api_key"])
    p.add_argument("--workers", type=int, default=CONFIG["workers"])
    p.add_argument("--temperature", type=float, default=CONFIG["temperature"])
    p.add_argument("--max_tokens", type=int, default=CONFIG["max_tokens"])
    p.add_argument("--no_resume", action="store_true", help="Ignore existing partial and start over")
    p.add_argument(
        "--hard_retry",
        action="store_true",
        help="Skip the unpenalized first try and use stronger anti-repetition sampling",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key or args.api_key.startswith("REPLACE_"):
        raise SystemExit("Set VOLCENGINE_ARK_API_KEY / ARK_API_KEY, or pass --api_key")

    all_data = json.loads(args.in_json.read_text(encoding="utf-8"))
    if not isinstance(all_data, list):
        raise SystemExit("input JSON must be a list")
    end = None if args.n <= 0 else args.offset + args.n
    slice_data = all_data[args.offset : end]
    if not slice_data:
        raise SystemExit("empty slice")

    k = rag_k(slice_data[0], args.with_rag)
    planned_n = len(slice_data)
    source_placeholder = f"{args.model}_rag{k}_n{planned_n}"

    partial_path = args.out_json.with_suffix(args.out_json.suffix + ".partial.json")
    if args.out_json.suffix == ".json":
        partial_path = args.out_json.with_name(args.out_json.stem + ".partial.json")

    done: dict[str, dict] = {} if args.no_resume else load_done(partial_path)
    todo = []
    for s in slice_data:
        sid = str(s.get("id"))
        if sid in done:
            continue
        todo.append(s)

    print(
        json.dumps(
            {
                "in_json": str(args.in_json),
                "out_json": str(args.out_json),
                "model": args.model,
                "base_url": args.base_url,
                "with_rag": args.with_rag,
                "rag_k": k,
                "slice_n": planned_n,
                "resume_done": len(done),
                "todo": len(todo),
                "workers": args.workers,
            },
            indent=2,
        ),
        flush=True,
    )

    if not todo:
        out = [done[str(s.get("id"))] for s in slice_data if str(s.get("id")) in done]
        source = f"{args.model}_rag{k}_n{len(out)}"
        for s in out:
            s["source"] = source
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"nothing to run; wrote {args.out_json} n={len(out)} source={source}", flush=True)
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=CONFIG["timeout"])
    errors: dict[str, str] = {}
    t0 = time.time()

    def job(sample: dict):
        sid = str(sample.get("id"))
        base_prompt = make_prompt(sample, args.with_rag)
        question = sample.get("question_with_options") or ""
        # Later attempts lock the original options and penalize repeated sentences.
        # Token cap stays high enough for a 2000–5000 character answer, low enough
        # that a repeated-line loop does not run to 8192.
        schedules = (
            (args.temperature, args.max_tokens, 0.0, 0.0, 1.0, False),
            (min(args.temperature, 0.4), min(args.max_tokens, 3072), 0.8, 0.4, 1.12, True),
            (0.3, min(args.max_tokens, 3072), 1.2, 0.8, 1.25, True),
            (0.2, min(args.max_tokens, 3072), 1.6, 1.2, 1.4, True),
            (0.15, min(args.max_tokens, 3072), 1.8, 1.5, 1.55, True),
        )
        if args.hard_retry:
            schedules = schedules[1:]
        last_err = "unknown"
        for attempt, (temp, max_tokens, presence, frequency, repetition, lock) in enumerate(schedules):
            prompt = base_prompt + (option_lock(question) if lock else "")
            try:
                text = call_model(
                    client,
                    args.model,
                    prompt,
                    temp,
                    max_tokens,
                    presence,
                    frequency,
                    repetition,
                )
                has_analysis = "**Analysis:**" in text or "Analysis:" in text
                has_conclusion = "**Conclusion:**" in text or "Conclusion:" in text
                if (not has_analysis or not has_conclusion) and len(text) >= 400 and not _is_repetition(text):
                    repaired = call_model(
                        client,
                        args.model,
                        repair_prompt(question, text),
                        0.2,
                        min(max_tokens, 3072),
                        0.6,
                        0.4,
                        1.15,
                    )
                    if "**Analysis:**" in repaired or "Analysis:" in repaired:
                        text = repaired
                if _is_repetition(text):
                    raise RuntimeError("repetition_loop")
                if len(text) < 400:
                    raise RuntimeError(f"too_short len={len(text)}")
                if "**Analysis:**" not in text and "Analysis:" not in text:
                    raise RuntimeError("missing Analysis")
                if "**Conclusion:**" not in text and "Conclusion:" not in text:
                    raise RuntimeError("missing Conclusion")
                if not re.search(
                    r"Therefore, the answer to the above question is: $",
                    text,
                ):
                    raise RuntimeError("missing bare therefore line")
                built = build_sample(
                    sample, text, args.model, args.with_rag, source_placeholder
                )
                return sid, built, None
            except Exception as e:
                last_err = str(e)
                time.sleep(2 * (attempt + 1))
        return sid, None, last_err

    def run_batch(samples: list[dict], label: str) -> None:
        if not samples:
            return
        print(f"{label}: {len(samples)} items", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(job, s): s for s in samples}
            finished = 0
            for fut in as_completed(futs):
                sid, built, err = fut.result()
                finished += 1
                if err:
                    errors[sid] = err
                    print(
                        f"[{finished}/{len(samples)}] FAIL id={sid} {err[:160]}",
                        flush=True,
                    )
                else:
                    errors.pop(sid, None)
                    done[sid] = built
                    print(
                        f"[{finished}/{len(samples)}] OK id={sid} "
                        f"intent={built['intent_letter']} gold={built.get('gold_letter')} "
                        f"done={len(done)}/{planned_n}",
                        flush=True,
                    )
                if len(done) % 20 == 0:
                    save_partial(
                        partial_path,
                        done,
                        {"source": source_placeholder, "ok": len(done), "todo": planned_n},
                    )
                    print(f"checkpoint {partial_path} n={len(done)}", flush=True)

    run_batch(todo, "pass1")
    retry_ids = set(errors)
    if retry_ids:
        retry_samples = [s for s in todo if str(s.get("id")) in retry_ids]
        print(f"retrying {len(retry_samples)} failures...", flush=True)
        run_batch(retry_samples, "pass2_retry")

    save_partial(
        partial_path,
        done,
        {"source": source_placeholder, "ok": len(done), "todo": planned_n},
    )

    missing = [str(s.get("id")) for s in slice_data if str(s.get("id")) not in done]
    if missing:
        print(
            f"incomplete: ok={len(done)} missing={len(missing)} partial={partial_path}",
            flush=True,
        )
        for sid in missing[:20]:
            print(sid, errors.get(sid, "not_done"), flush=True)
        sys.exit(1)

    out = [done[str(s.get("id"))] for s in slice_data]
    source = f"{args.model}_rag{k}_n{len(out)}"
    for s in out:
        s["source"] = source

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    both = [s for s in out if s.get("gold_letter") in list("ABCDE") and s.get("intent_letter") in list("ABCDE")]
    agree = sum(1 for s in both if s["gold_letter"] == s["intent_letter"])
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "source": source,
                "n": len(out),
                "with_rag": args.with_rag,
                "rag_k": k,
                "elapsed_sec": round(time.time() - t0, 1),
                "intent_vs_gold_agree": agree,
                "intent_vs_gold_n": len(both),
                "intent_vs_gold_rate": round(agree / max(len(both), 1), 4),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
