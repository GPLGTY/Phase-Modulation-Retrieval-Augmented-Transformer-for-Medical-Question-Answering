#!/usr/bin/env python3
"""
FieldNet Length Extrapolation Evaluation - TNNLS Standard (Fixed + Speed Test)
FieldNet 长度外推评估 - TNNLS 标准（固定+速度测试）
"""

import os
import sys
import time
import random
import warnings
import argparse
import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# 评估配置数据类 | Evaluation Configuration Dataclass
@dataclass
class EvalConfig:
    model_path: str
    device: str = "cuda:0"
    batch_size: int = 4
    stride: int = 512
    max_samples: int = 50
    seeds: List[int] = None
    output_dir: str = "./eval_results"
    lang: str = "en"
    pg19_path: str = "./data/p19"
    chinese_novel_path: str = "./data/chinese_novel/let_him_land.txt"
    seq_lengths: List[int] = None
    strict_numerical: bool = True
    max_position_embeddings: int = None

    def __post_init__(self):
        # 初始化默认参数 | Initialize default parameters
        if self.seeds is None:
            self.seeds = [42, 123, 456]
        if self.seq_lengths is None:
            self.seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
        if self.max_position_embeddings is None:
            self.max_position_embeddings = max(self.seq_lengths)


# 设置随机种子确保实验可复现 | Set random seed for experiment reproducibility
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# 加载本地PG19数据集 | Load local PG19 dataset
def load_local_pg19(
    data_dir: str = "./data/p19", seq_lengths: List[int] = None
) -> List[str]:
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"未找到 {data_dir}/test-*.parquet | Not found {data_dir}/test-*.parquet"
        )

    ds = load_dataset("parquet", data_files=str(parquet_files[0]))["train"]
    texts = []
    for item in ds:
        text = item.get("text", "")
        if len(text) >= 100_000:
            texts.append(text[:500_000])
            if len(texts) >= 100:
                break

    # 按文本长度降序排序 | Sort texts by length in descending order
    texts.sort(key=lambda x: len(x), reverse=True)
    logger.info(
        f"加载 {len(texts)} 本书，最长: {len(texts[0]):,} 字符 | Loaded {len(texts)} books, max length: {len(texts[0]):,} chars"
    )
    return texts


# 加载本地中文小说数据集 | Load local Chinese novel dataset
def load_local_chinese_novel(
    file_path: str = "./data/chinese_novel/let_him_land.txt",
) -> List[str]:
    file_path = Path(file_path)
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    text = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
                break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("无法解码文件 | Failed to decode file")
    logger.info(
        f"加载中文小说: {len(text):,} 字符 | Loaded Chinese novel: {len(text):,} chars"
    )
    return [text]


# 加载模型与分词器 | Load model and tokenizer
def load_model_and_tokenizer(model_path: str, force_max_len: int = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 强制设置最大位置编码长度 | Force set max position embedding length
    if force_max_len is not None:
        setattr(config, "max_position_embeddings", force_max_len)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 设置填充token | Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    # model = torch.compile(model, mode="default")
    return model, tokenizer


# 评估困惑度与推理速度 | Evaluate perplexity and inference speed
def evaluate_perplexity_and_speed(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    seq_len: int,
    device: torch.device,
    batch_size: int = 4,
    stride: int = 512,
    max_samples: int = 50,
) -> Tuple[Dict[str, float], Dict]:
    """
    核心评估函数 | Core evaluation function
    """
    model.eval()
    # 重要：Loss function 不 reduction='mean'，先 sum 再除，保持精度
    # Important: Use sum reduction for loss to maintain precision
    loss_fn = nn.CrossEntropyLoss(reduction="sum", ignore_index=tokenizer.pad_token_id)

    # --- 1. 数据准备 (单本书循环拼接) --- | Data preparation (single book loop concatenation)
    primary_text = texts[0]
    single_book_ids = tokenizer.encode(primary_text, add_special_tokens=False)

    target_length = seq_len * 10
    all_ids = single_book_ids.copy()
    while len(all_ids) < target_length:
        all_ids.extend(single_book_ids)

    logger.info(f"[Data] 总Tokens: {len(all_ids):,} | Total tokens: {len(all_ids):,}")

    # --- 2. 构建样本 --- | Build samples
    dataset = []
    max_start = len(all_ids) - seq_len - 1
    starts = list(range(0, max_start + 1, stride))
    random.shuffle(starts)
    starts = starts[:max_samples]

    for start in starts:
        end = start + seq_len + 1
        chunk = all_ids[start:end]
        if len(chunk) < seq_len + 1:
            chunk += [tokenizer.pad_token_id] * (seq_len + 1 - len(chunk))
        dataset.append(torch.tensor(chunk, dtype=torch.long))

    # --- 3. 构建 Batch --- | Build batches
    dataset.sort(key=lambda x: len(x))
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        max_len_b = max(len(x) for x in batch)
        padded = torch.stack(
            [
                (
                    torch.cat(
                        [
                            x,
                            torch.full(
                                (max_len_b - len(x),),
                                tokenizer.pad_token_id,
                                dtype=torch.long,
                            ),
                        ]
                    )
                    if len(x) < max_len_b
                    else x
                )
                for x in batch
            ]
        )
        batches.append(padded)

    # --- 4. 评估循环 --- | Evaluation loop
    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0
    has_warmed_up = False

    with torch.no_grad():
        for batch_tensor in tqdm(batches, desc=f"Len={seq_len}"):
            batch_tensor = batch_tensor.to(device)
            input_ids = batch_tensor[:, :-1].contiguous()
            target_ids = batch_tensor[:, 1:].contiguous()
            attn_mask = (input_ids != tokenizer.pad_token_id).long()

            # 测速开始 | Start timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.time()

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attn_mask, use_cache=False
                )
                logits = outputs.logits

            # 测速结束 | End timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_t = time.time() - start_t

            # 第一个batch预热不计入时间 | Skip first batch for warm-up
            if has_warmed_up:
                total_time += elapsed_t
            else:
                has_warmed_up = True

            # --- 关键修复：使用 Float32 计算 Loss --- | Critical fix: Use Float32 for loss calculation
            # logits 是 float16，必须转 float32 否则长序列容易溢出 inf
            # logits are float16, must convert to float32 to avoid overflow in long sequences
            logits = logits.float()

            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            valid_tokens = (target_ids != tokenizer.pad_token_id).sum().item()

            if valid_tokens > 0:
                total_loss += loss.item()
                total_tokens += valid_tokens

    if total_tokens == 0:
        raise RuntimeError("无有效Token | No valid tokens")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))  # 防止 exp 溢出 | Prevent exp overflow

    # 计算每秒token数 | Calculate tokens per second
    if total_time > 0:
        tps = total_tokens / total_time
    else:
        tps = 0

    return {"ppl": ppl, "tps": tps}, {"avg_loss": avg_loss}


# 运行单个种子的评估 | Run evaluation with single seed
def run_single_seed(config: EvalConfig, seed: int) -> Dict[int, Dict]:
    setup_seed(seed)

    # 根据语言加载数据集 | Load dataset based on language
    if config.lang == "zh":
        texts = load_local_chinese_novel(config.chinese_novel_path)
    else:
        texts = load_local_pg19(config.pg19_path)

    results = {}
    for seq_len in config.seq_lengths:
        logger.info(f"\n评估长度: {seq_len} | Evaluating sequence length: {seq_len}")

        model, tokenizer = load_model_and_tokenizer(
            config.model_path, force_max_len=seq_len
        )
        model = model.to(config.device)

        # 根据序列长度动态调整batch大小 | Dynamically adjust batch size by sequence length
        if seq_len == 1024:
            eff_batch = 32
        elif seq_len == 2048:
            eff_batch = 10
        elif seq_len == 4096:
            eff_batch = 6
        elif seq_len == 8192:
            eff_batch = 2
        else:  # 16384, 32768
            eff_batch = 1

        logger.info(
            f"应用配置: Sequence Length={seq_len}, Batch Size={eff_batch} | Applied config: seq_len={seq_len}, batch_size={eff_batch}"
        )
        # ==========================================

        try:
            metrics, meta = evaluate_perplexity_and_speed(
                model,
                tokenizer,
                texts,
                seq_len,
                torch.device(config.device),
                eff_batch,
                config.stride,
                config.max_samples,
            )
            results[seq_len] = metrics
            logger.info(
                f"✓ PPL: {metrics['ppl']:.2f} | Speed: {metrics['tps']:.2f} tokens/s | Loss: {meta['avg_loss']:.4f}"
            )
        except Exception as e:
            logger.error(f"✗ 失败: {e} | Failed: {e}")
            results[seq_len] = {"ppl": float("inf"), "tps": 0}

        # 清理显存 | Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    return results


# 主函数 | Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lang", default="en", type=str)
    parser.add_argument("--pg19_path", default="./data/p19", type=str)
    args = parser.parse_args()

    # 解析模型参数 | Parse model arguments
    model_path = None
    for arg in args.model_args.split(","):
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k == "pretrained":
                model_path = v.strip()

    config = EvalConfig(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        lang=args.lang,
        pg19_path=args.pg19_path,
    )

    # 运行多组种子评估 | Run evaluation with multiple seeds
    all_results = []
    for seed in config.seeds:
        try:
            res = run_single_seed(config, seed)
            all_results.append(res)
        except Exception as e:
            logger.error(f"种子 {seed} 失败: {e} | Seed {seed} failed: {e}")

    # 汇总并打印结果 | Summarize and print results
    if all_results:
        print("\n最终结果 (种子平均): | Final results (average over seeds):")
        for seq_len in config.seq_lengths:
            ppls = [
                r[seq_len]["ppl"]
                for r in all_results
                if math.isfinite(r[seq_len]["ppl"])
            ]
            tpss = [r[seq_len]["tps"] for r in all_results if r[seq_len]["tps"] > 0]
            if ppls:
                print(
                    f"Len {seq_len:<6} -> PPL: {np.mean(ppls):.2f} (±{np.std(ppls):.2f}), TPS: {np.mean(tpss):.2f}"
                )


if __name__ == "__main__":
    main()
# python extrapolation.py --lang zh --model_args 'pretrained=./MiniMind_Base_300m,trust_remote_code=True'  --pg19_path  './data/chinese_novel'
# python extrapolation.py --lang en --model_args 'pretrained=./QF_Base_300M,trust_remote_code=True'  --pg19_path  './data/pg19'
