#!/usr/bin/env python3
"""
temperature（QubitEmbed）× ntk_alpha（NTK RoPE）推理期消融。

固定权重，在 dev intent JSON 上批量推理，输出与 inference_intent_results_s1024.csv 同列的逐条 CSV，
并汇总 accuracy 到 results/ablation/summary.csv。

用法（项目根目录）：

  python scripts/run_ablation_temp_ntk.py
  python scripts/run_ablation_temp_ntk.py --limit 32   # 冒烟
  python scripts/run_ablation_temp_ntk.py --batch_size 4
  python scripts/run_ablation_temp_ntk.py --temperatures 0.9,1.1,1.3 --ntk_alphas 0.5,0.8,1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

CSV_FIELDS = [
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

DEFAULT_TEMPERATURES = [0.7, 0.9, 1.1, 1.3, 1.5]
DEFAULT_NTK_ALPHAS =  [0.4, 0.6, 0.8, 1.0, 1.2]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _param_tag(name: str, value: float) -> str:
    """文件名友好：temp1p10 / ntk0p80"""
    return f"{name}{str(value).replace('.', 'p')}"


def apply_ablation_settings(
    model: QubitFormer,
    temperature: float,
    ntk_alpha: float,
    ntk_max_train_len: int,
) -> None:
    model.tok_embeddings.temperature = float(temperature)
    model.params.ntk_alpha = float(ntk_alpha)
    model.dynamic_ntk.alpha = float(ntk_alpha)
    model.dynamic_ntk.max_train_len = int(ntk_max_train_len)
    for layer in model.layers:
        layer.attn.dynamic_ntk.alpha = float(ntk_alpha)
        layer.attn.dynamic_ntk.max_train_len = int(ntk_max_train_len)


def load_model_and_tokenizer(args, device: torch.device):
    lm_config = QFConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
        vocab_size=32000,
        n_kv_heads=args.num_key_value_heads,
        ntk_alpha=0.8,
    )
    tok_dir = ROOT / "model" / "tokenizer_en"
    tokenizer = LlamaTokenizerFast.from_pretrained(str(tok_dir))
    # Llama 默认常无 pad_token；若 pad_token_id 落在 <unk>=0，批 padding 会写满 0，
    # CSV 里 decode(input_ids[:sum(mask)]) 又会误取「左侧填充区」→ 一大片 <unk>。
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    model = QubitFormer(lm_config)
    state = torch.load(args.checkpoint, map_location="cpu")
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

    return model, tokenizer, lm_config


def _row_skeleton(item: dict) -> dict:
    return {
        "id": item.get("id", ""),
        "meta_info": item.get("meta_info", ""),
        "question_with_options": item.get("question_with_options", ""),
        "assistant_content": "",
        "answer_with_idx": item.get("answer_with_idx", ""),
        "answer_idx": (item.get("answer_idx") or "").strip().upper(),
        "model_input_prompt": "",
        "model_raw_output": "",
        "extracted_choice": "",
        "matches_answer_idx": False,
        "error": "",
    }


def _prepare_record(item: dict) -> dict:
    row = _row_skeleton(item)
    system_txt, user_txt, assistant_txt = parse_messages(item.get("messages", []))
    row["assistant_content"] = (assistant_txt or "").strip()
    if not user_txt:
        row["error"] = "messages 中缺少 role=user"
        return {"row": row, "prompt": None}
    merged = merge_system_and_user_body(system_txt, user_txt)
    return {"row": row, "prompt": format_inst_prompt_sft(merged)}


def _build_gen_kw(args, tokenizer) -> dict:
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
    return gen_kw


def _decode_one_output(
    tokenizer,
    input_ids_row,
    output_ids_row,
    attention_mask_row,
    padded_input_width: int,
    assistant_content: str,
    args,
) -> tuple[str, str, str, bool]:
    """从 generate 结果中拆出 prompt 与生成段。

    ``padded_input_width`` 为 **左填充后** 输入序列在 ``output_ids_row`` 中所占长度
    （与 ``tokenizer(..., padding=True)`` 的 ``input_ids.shape[1]`` 一致），不要用
    ``attention_mask.sum()`` 代替，否则会切错新生成 token。
    CSV 里写回 ``model_input_prompt`` 时仅用 ``attention_mask`` 为 1 的位置 decode，避免解码 padding。
    """
    if args.no_csv_model_input:
        model_input_prompt = ""
    else:
        valid = input_ids_row[attention_mask_row.bool()]
        model_input_prompt = tokenizer.decode(valid, skip_special_tokens=False)

    gen_ids = output_ids_row[padded_input_width:]
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
    return model_input_prompt, out_text, ex, ex


def _infer_single(
    model,
    tokenizer,
    record: dict,
    args,
    device: torch.device,
    max_input_len: int,
    autocast_ctx,
    gen_kw: dict,
) -> dict:
    row = dict(record["row"])
    prompt = record["prompt"]
    try:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        padded_w = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            with autocast_ctx():
                out = model.generate(**enc, **gen_kw)
        mip, out_text, ex, _ = _decode_one_output(
            tokenizer,
            enc["input_ids"][0].cpu(),
            out[0].cpu(),
            enc["attention_mask"][0].cpu(),
            padded_w,
            row["assistant_content"],
            args,
        )
        row["model_input_prompt"] = mip
        row["model_raw_output"] = out_text
        row["extracted_choice"] = ex
        row["matches_answer_idx"] = bool(ex) and (ex == row["answer_idx"])
    except Exception as e:
        row["error"] = str(e)
        row["matches_answer_idx"] = False
    return row


def _infer_batch(
    model,
    tokenizer,
    batch_records: list,
    args,
    device: torch.device,
    max_input_len: int,
    autocast_ctx,
    gen_kw: dict,
) -> list[dict]:
    prompts = [r["prompt"] for r in batch_records]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    padded_w = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        with autocast_ctx():
            out = model.generate(**enc, **gen_kw)

    rows = []
    for i, rec in enumerate(batch_records):
        row = dict(rec["row"])
        mip, out_text, ex, _ = _decode_one_output(
            tokenizer,
            enc["input_ids"][i].cpu(),
            out[i].cpu(),
            enc["attention_mask"][i].cpu(),
            padded_w,
            row["assistant_content"],
            args,
        )
        row["model_input_prompt"] = mip
        row["model_raw_output"] = out_text
        row["extracted_choice"] = ex
        row["matches_answer_idx"] = bool(ex) and (ex == row["answer_idx"])
        rows.append(row)
    return rows


def run_one_combo(
    model: QubitFormer,
    tokenizer: LlamaTokenizerFast,
    data: list,
    args,
    device: torch.device,
    temperature: float,
    ntk_alpha: float,
    output_csv: Path,
) -> dict:
    apply_ablation_settings(
        model, temperature, ntk_alpha, args.ntk_max_train_len
    )

    max_input_len = max(64, args.max_seq_len - args.max_new_tokens - 8)
    batch_size = max(1, int(args.batch_size))
    gen_kw = _build_gen_kw(args, tokenizer)

    if device.type != "cuda" or args.dtype == "float32":
        autocast_ctx = nullcontext
    elif args.dtype == "bfloat16":

        def autocast_ctx():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)

    else:

        def autocast_ctx():
            return torch.cuda.amp.autocast(dtype=torch.float16)

    records = [_prepare_record(item) for item in data]
    n_ok = 0
    n_match = 0
    n_err = 0
    t0 = time.time()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        n_batches = (len(records) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(records), batch_size), total=n_batches, leave=False)

        for start in pbar:
            pbar.set_description(
                f"T={temperature} ntk={ntk_alpha} bs={batch_size}"
            )
            chunk = records[start : start + batch_size]
            invalid = [r for r in chunk if r["prompt"] is None]
            valid = [r for r in chunk if r["prompt"] is not None]

            for r in invalid:
                writer.writerow(r["row"])
                n_err += 1

            if not valid:
                continue

            try:
                if len(valid) == 1:
                    rows = [
                        _infer_single(
                            model,
                            tokenizer,
                            valid[0],
                            args,
                            device,
                            max_input_len,
                            autocast_ctx,
                            gen_kw,
                        )
                    ]
                else:
                    rows = _infer_batch(
                        model,
                        tokenizer,
                        valid,
                        args,
                        device,
                        max_input_len,
                        autocast_ctx,
                        gen_kw,
                    )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rows = []
                for rec in valid:
                    try:
                        rows.append(
                            _infer_single(
                                model,
                                tokenizer,
                                rec,
                                args,
                                device,
                                max_input_len,
                                autocast_ctx,
                                gen_kw,
                            )
                        )
                    except Exception as e2:
                        row = dict(rec["row"])
                        row["error"] = str(e2)
                        rows.append(row)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            for row in rows:
                writer.writerow(row)
                if row.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                    if row.get("matches_answer_idx"):
                        n_match += 1

    elapsed = time.time() - t0
    n_total = len(data)
    acc = (n_match / n_ok) if n_ok else 0.0
    return {
        "temperature": temperature,
        "ntk_alpha": ntk_alpha,
        "ntk_max_train_len": args.ntk_max_train_len,
        "batch_size": batch_size,
        "n_total": n_total,
        "n_scored": n_ok,
        "n_match": n_match,
        "n_error": n_err,
        "accuracy": acc,
        "elapsed_sec": round(elapsed, 1),
        "output_csv": str(output_csv),
    }


def main():
    parser = argparse.ArgumentParser(
        description="temperature × ntk_alpha 推理消融（dev intent）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "out" / "run_intent_seq8192_20260521" / "sft_intent_QF_en_1024.pth"),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "dataset" / "RAG_MedQA_US_dev_1272_intent.json"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "results" / "ablation"),
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TEMPERATURES),
    )
    parser.add_argument(
        "--ntk_alphas",
        type=str,
        default=",".join(str(a) for a in DEFAULT_NTK_ALPHAS),
    )
    parser.add_argument(
        "--ntk_max_train_len",
        type=int,
        default=1024,
        help="NTK 触发阈值；默认 1024，使 8192 推理下 ntk_alpha 生效",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float32")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=24)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="推理 micro-batch；OOM 时自动对该 batch 逐条重试",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-csv-model-input", action="store_true")
    parser.add_argument("--no-compact-model-output", action="store_true")
    parser.add_argument(
        "--extract_choice_from",
        choices=("model", "assistant", "both"),
        default="model",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若对应 CSV 已存在则跳过该组（便于断点续跑）",
    )
    args = parser.parse_args()

    temperatures = _parse_float_list(args.temperatures)
    ntk_alphas = _parse_float_list(args.ntk_alphas)
    if not temperatures or not ntk_alphas:
        print("错误: temperatures / ntk_alphas 不能为空", file=sys.stderr)
        sys.exit(1)

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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"检查点: {ckpt}", flush=True)
    print(f"数据: {data_path} | n={len(data)}", flush=True)
    print(f"网格: |T|={len(temperatures)} |ntk|={len(ntk_alphas)}", flush=True)
    print(f"NTK max_train_len={args.ntk_max_train_len} | max_seq_len={args.max_seq_len}", flush=True)
    print(f"batch_size={max(1, args.batch_size)}", flush=True)

    model, tokenizer, lm_config = load_model_and_tokenizer(args, device)

    manifest = {
        "checkpoint": ckpt,
        "data_path": data_path,
        "n_samples": len(data),
        "temperatures": temperatures,
        "ntk_alphas": ntk_alphas,
        "ntk_max_train_len": args.ntk_max_train_len,
        "max_seq_len": args.max_seq_len,
        "batch_size": max(1, args.batch_size),
        "lm_config": {
            "dim": lm_config.dim,
            "n_layers": lm_config.n_layers,
            "n_heads": lm_config.n_heads,
            "n_kv_heads": lm_config.n_kv_heads,
        },
        "runs": [],
    }

    summary_path = results_dir / "summary.csv"
    summary_fields = [
        "run_id",
        "temperature",
        "ntk_alpha",
        "ntk_max_train_len",
        "batch_size",
        "n_total",
        "n_scored",
        "n_match",
        "n_error",
        "accuracy",
        "elapsed_sec",
        "output_csv",
    ]

    run_id = 0
    all_rows = []

    for temperature in temperatures:
        for ntk_alpha in ntk_alphas:
            run_id += 1
            fname = (
                f"intent_dev_{_param_tag('temp', temperature)}_"
                f"{_param_tag('ntk', ntk_alpha)}.csv"
            )
            out_csv = results_dir / fname

            if args.skip_existing and out_csv.is_file():
                print(f"[{run_id}] 跳过已存在: {out_csv}", flush=True)
                continue

            print(
                f"[{run_id}/{len(temperatures)*len(ntk_alphas)}] "
                f"temperature={temperature} ntk_alpha={ntk_alpha} -> {out_csv.name}",
                flush=True,
            )
            row = run_one_combo(
                model,
                tokenizer,
                data,
                args,
                device,
                temperature,
                ntk_alpha,
                out_csv,
            )
            row["run_id"] = run_id
            all_rows.append(row)
            manifest["runs"].append(row)
            print(
                f"    accuracy={row['accuracy']:.4f} "
                f"({row['n_match']}/{row['n_scored']}) "
                f"time={row['elapsed_sec']}s",
                flush=True,
            )

            with open(summary_path, "w", encoding="utf-8-sig", newline="") as sf:
                w = csv.DictWriter(sf, fieldnames=summary_fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rows)

    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"\n汇总: {summary_path}", flush=True)
    print(f"配置: {manifest_path}", flush=True)
    if all_rows:
        best = max(all_rows, key=lambda r: r["accuracy"])
        print(
            f"最高 accuracy={best['accuracy']:.4f} @ "
            f"T={best['temperature']} ntk={best['ntk_alpha']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
