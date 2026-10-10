#!/usr/bin/env python3
"""
在仓库内自定义 QubitFormer 检查点（如 train_sft_intent 产出的 sft_intent_QF_en_512.pth）上，
对 MedQA **intent** JSON（messages 含 system / user / assistant）批量推理并写 CSV。

**输入模型侧**：仅将 `messages` 中 **system + user** 合并后包进 `<s>[INST] … [/INST]`（与二阶段 SFT 训练一致），
**不**把 `role=assistant` 的 content 喂给模型；CSV 仍写入 `assistant_content` 列便于对照标签。

列格式对齐 `inference_results.csv`（id, meta_info, question_with_options, assistant_content, …）。

用法（项目根目录）：

  python inference_intent_qf_csv.py \\
    --checkpoint out/sft_intent_QF_en_512.pth \\
    --data_path dataset/RAG_MedQA_US_test_1273_intent.json \\
    --output_csv inference_intent_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from inference import (
    decode_generation_texts,
    format_inst_prompt_sft,
    merge_system_and_user_body,
    parse_messages,
    postprocess_model_raw_output,
    resolve_extracted_choice,
)
from model.QFConfig import QFConfig
from model.qbmodel import QubitFormer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Intent SFT checkpoint → CSV (system+user only)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "out" / "sft_intent_QF_en_512.pth"),
        help="sft_intent_*.pth 权重路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "dataset" / "RAG_MedQA_US_test_1273_intent.json"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ROOT / "inference_intent_results.csv"),
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="与训练时 QFConfig.max_seq_len 一致")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--min_new_tokens", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="仅跑前 N 条，0 为全部")
    parser.add_argument(
        "--no-csv-model-input",
        action="store_true",
        help="不写入 model_input_prompt 列（置空），减小 CSV",
    )
    parser.add_argument(
        "--no-compact-model-output",
        action="store_true",
        help="不对 model_raw_output 做 postprocess（与 inference.py --no-compact-model-output 一致）",
    )
    parser.add_argument(
        "--extract_choice_from",
        choices=("model", "assistant", "both"),
        default="model",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        print(f"错误: 找不到检查点: {ckpt}", file=sys.stderr)
        sys.exit(1)

    data_path = os.path.abspath(args.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("错误: JSON 根应为数组", file=sys.stderr)
        sys.exit(1)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    vocab_size = 32000
    lm_config = QFConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
        vocab_size=vocab_size,
        n_kv_heads=args.num_key_value_heads,
    )

    tok_dir = ROOT / "model" / "tokenizer_en"
    tokenizer = LlamaTokenizerFast.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = QubitFormer(lm_config)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    if device.type == "cpu":
        for m in model.modules():
            if m.__class__.__name__ == "QubitAttention":
                m.flash = False

    if args.dtype == "bfloat16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    elif args.dtype == "float16" and device.type == "cuda":
        model = model.to(dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)

    max_input_len = max(64, args.max_seq_len - args.max_new_tokens - 8)
    print(
        f"加载: {ckpt} | 样本数: {len(data)} | max_input_len={max_input_len} | max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    out_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
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

    def autocast_ctx():
        if device.type != "cuda" or args.dtype == "float32":
            from contextlib import nullcontext

            return nullcontext()
        if args.dtype == "bfloat16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        fout.flush()

        for item in tqdm(data, desc="intent_infer"):
            rec_id = item.get("id", "")
            meta = item.get("meta_info", "")
            qwo = item.get("question_with_options", "")
            awi = item.get("answer_with_idx", "")
            aidx = (item.get("answer_idx") or "").strip().upper()
            err = ""

            system_txt, user_txt, assistant_txt = parse_messages(item.get("messages", []))
            assistant_content = (assistant_txt or "").strip()

            if not user_txt:
                err = "messages 中缺少 role=user"
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
                        "error": err,
                    }
                )
                fout.flush()
                continue

            merged = merge_system_and_user_body(system_txt, user_txt)
            prompt = format_inst_prompt_sft(merged)

            try:
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_len,
                    add_special_tokens=False,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                in_len = int(enc["input_ids"].shape[1])
                if args.no_csv_model_input:
                    model_input_prompt = ""
                else:
                    model_input_prompt = tokenizer.decode(
                        enc["input_ids"][0], skip_special_tokens=False
                    )

                gen_kw = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                if args.min_new_tokens and args.min_new_tokens > 0:
                    gen_kw["min_new_tokens"] = min(args.min_new_tokens, args.max_new_tokens)

                with torch.no_grad():
                    with autocast_ctx():
                        out = model.generate(**enc, **gen_kw)
                gen_ids = out[0, in_len:]
                raw_extract, raw_csv = decode_generation_texts(tokenizer, gen_ids)
                canonical = raw_csv if not (raw_extract or "").strip() else raw_extract
                out_text = (
                    postprocess_model_raw_output(canonical)
                    if not args.no_compact_model_output
                    else canonical
                )
                ex = resolve_extracted_choice(
                    assistant_content, out_text, args.extract_choice_from
                )
                match = bool(ex) and (ex == aidx)
            except Exception as e:
                err = str(e)
                model_input_prompt = ""
                out_text = ""
                ex = ""
                match = False

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
                    "error": err,
                }
            )
            fout.flush()

    print(f"已保存: {out_path}", flush=True)


if __name__ == "__main__":
    main()
