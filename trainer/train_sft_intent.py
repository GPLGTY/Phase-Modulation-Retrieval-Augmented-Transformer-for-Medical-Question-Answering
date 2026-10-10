"""
二阶段 SFT：在已有英文 SFT 权重上，用 `RAG_MedQA_US_*_intent.json` 的 messages 继续全参微调。

训练过程中默认写出两类 checkpoint：``{stem}.pth`` 为最新快照（``save_interval`` 与训练结束）；
``{stem}_best.pth`` 在未指定 ``--val_data_path`` 时为「训练 masked CE batch」历史最低；
若指定验证集则默认按 **验证集选项首 token 准确率**（``--best_metric val_answer_acc``）保存 best，
亦可改为 ``val_loss``。loss 仅在 [/INST] 后非 pad 的 token 上计算（不含 padding 区）。
每个 epoch 验证结束会在 ``{out_dir}/val_results/`` 写出与 ``results/ablation_p3_2/intent_dev_*.csv``
同列的逐条 CSV（``val_epoch01.csv`` …；刷新 best 时另写 ``val_best.csv``）。可用 ``--no_val_results_csv`` 关闭。

默认针对单卡 NVIDIA A800（80GB）：较长上下文 + 适中 micro-batch，配合梯度累积放大等效 batch。

用法示例（在项目根目录执行）：

初始化权重建议使用官方英文指令微调 **QF_Struct_300M** 导出的 Torch 检查点 ``parameters/sft_QF_en_1024.pth``（若尚无该文件：先放好 ``parameters/QF_Struct_300M``，再执行 ``scripts/stage2_intent_from_QF_Struct.sh`` 前半段导出逻辑，或使用 ``tools/convert_model.py`` 按该脚本同款参数转换）。下文模型规模与 ``parameters/QF_Struct_300M/config.json`` 一致。

默认 ``--early_stop_loss=0``（关闭）；若开启，阈值应设在 **去掉 pad 后的 answer CE** 上（单字母标签时 CE 也不会像含 pad 时那样虚低）。

单行命令（建议整段复制到 bash）::

  python trainer/train_sft_intent.py --init_ckpt parameters/sft_QF_en_1024.pth --data_path dataset/RAG_MedQA_US_train_10178_intent.json --out_dir out/run_sft_False_False --model_type QF --language en --hidden_size 1024 --num_hidden_layers 24 --num_attention_heads 16 --num_key_value_heads 4 --max_seq_len 4096 --batch_size 3 --accumulation_steps 8 --epochs 6 --sft_lr 5e-5 --early_stop_loss 0 --val_data_path dataset/RAG_MedQA_US_dev_1272_intent.json

若必须手写多续行：行末只放一个反斜杠续行符，该行反斜杠后不能有空格（也不要在本文件字符串里写成连续两个 ``\\`` 再整段粘贴到终端）。

多卡（torchrun）：在单行命令的参数列表前加上 ``torchrun --nproc_per_node=8 trainer/train_sft_intent.py --ddp``，其余 ``--xxx`` 与上相同。
"""

from __future__ import annotations

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import math
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import LlamaTokenizerFast

import logging

from model.QFConfig import QFConfig
from model.model import MiniMindForCausalLM
from model.qbmodel import QubitFormer
from dataset.lm_en_dataset import SFTIntentMessagesDataset
from inference import format_inst_prompt_sft, merge_system_and_user_body

warnings.filterwarnings("ignore")

VAL_CSV_FIELDS = [
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

logger = None
ddp = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logger(log_file: str):
    global logger
    logger = logging.getLogger("SFTIntent")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def Logger(content: str):
    if not ddp or dist.get_rank() == 0:
        if logger:
            logger.info(content)
        else:
            print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def checkpoint_stem() -> str:
    return f"sft_intent_{args.model_type}_{args.language}_{lm_config.dim}"


def masked_ce_batch(
    logits: torch.Tensor, Y: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """在 loss_mask=1 的位置上求 token CE 平均；返回 (标量 loss, 监督 token 数)。"""
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    lp = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())
    den = int(loss_mask.sum().item())
    if den <= 0:
        return logits.new_tensor(float("nan")), 0
    return (lp * loss_mask).sum() / loss_mask.sum(), den


def _val_row_skeleton(item: dict) -> dict:
    """与 ``scripts/run_ablation_temp_ntk.py`` 输出 CSV 列一致。"""
    system_txt, user_txt, assistant_txt = SFTIntentMessagesDataset._parse_messages(
        item.get("messages", [])
    )
    merged = merge_system_and_user_body(system_txt, user_txt) if user_txt else ""
    return {
        "id": item.get("id", ""),
        "meta_info": item.get("meta_info", ""),
        "question_with_options": item.get("question_with_options", ""),
        "assistant_content": (assistant_txt or "").strip(),
        "answer_with_idx": item.get("answer_with_idx", ""),
        "answer_idx": (item.get("answer_idx") or "").strip().upper(),
        "model_input_prompt": format_inst_prompt_sft(merged) if merged else "",
        "model_raw_output": "",
        "extracted_choice": "",
        "matches_answer_idx": False,
        "error": "",
    }


def _fill_val_prediction(row: dict, pred_token_id: int, tokenizer) -> None:
    """用 [/INST] 后首 token 的 argmax 填充 model_raw_output / extracted_choice。"""
    raw = tokenizer.decode([pred_token_id], skip_special_tokens=False)
    row["model_raw_output"] = raw
    extracted = ""
    text = raw.strip()
    if text:
        ch = text[0].upper()
        if ch in "ABCDE":
            extracted = ch
    row["extracted_choice"] = extracted
    ans = row["answer_idx"]
    row["matches_answer_idx"] = bool(extracted and ans and extracted == ans)


def write_val_results_csv(rows: list[dict], csv_path: str) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=VAL_CSV_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def eval_val_predictions(
    val_ds: SFTIntentMessagesDataset,
    val_dl: DataLoader,
    pad_token_id: int | None,
) -> tuple[float, int, list[dict]]:
    """
    验证集上统计：[/INST] 后第一个非 pad 监督 token（选项字母）的 argmax 准确率，
    并收集与 ablation CSV 同列的逐条结果。
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    rows: list[dict] = []
    sample_idx = 0
    try:
        for X, Y, loss_mask in val_dl:
            X = X.to(args.device, non_blocking=True)
            Y = Y.to(args.device, non_blocking=True)
            loss_mask = loss_mask.to(args.device, non_blocking=True)
            with ctx:
                with torch.cuda.amp.autocast(enabled=(args.dtype != "float32")):
                    res = model(X)
                    logits = res.logits.float()
            for b in range(Y.size(0)):
                item = val_ds.samples[sample_idx]
                sample_idx += 1
                row = _val_row_skeleton(item)
                try:
                    sup = loss_mask[b] > 0
                    if pad_token_id is not None:
                        sup = sup & (Y[b] != pad_token_id)
                    idxs = sup.nonzero(as_tuple=False).flatten()
                    if idxs.numel() == 0:
                        row["error"] = "无监督 token"
                        rows.append(row)
                        continue
                    t0 = int(idxs[0].item())
                    pred = int(logits[b, t0].argmax().item())
                    gold = int(Y[b, t0].item())
                    _fill_val_prediction(row, pred, tokenizer)
                    total += 1
                    if pred == gold:
                        correct += 1
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)
    finally:
        if was_training:
            model.train()
    if total <= 0:
        return float("nan"), 0, rows
    return correct / total, total, rows


def save_checkpoint(kind: str = "latest") -> str:
    """保存当前权重到 save_dir。

    kind:
      - ``latest``: 文件名 ``{stem}.pth``（总是当前训练状态下的最新快照）
      - ``best``:   文件名 ``{stem}_best.pth``（无验证集时按训练 batch loss；有 ``--val_data_path`` 时按验证损失）

    非主进程返回空串且不写盘。
    """
    if kind not in ("latest", "best"):
        raise ValueError(f"save_checkpoint kind must be 'latest' or 'best', got {kind!r}")
    if ddp and dist.get_rank() != 0:
        return ""
    model.eval()
    stem = checkpoint_stem()
    weight_filename = f"{stem}.pth" if kind == "latest" else f"{stem}_best.pth"
    ckp = os.path.join(args.save_dir, weight_filename)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    state_dict = {k: v.half() for k, v in raw_model.state_dict().items()}
    torch.save(state_dict, ckp)
    model.train()
    return ckp


@torch.no_grad()
def compute_val_masked_ce_mean(val_dl: DataLoader) -> float:
    """验证集上对 masked CE 做 token 加权的全局平均（仅非 pad 监督位，与训练一致）。"""
    was_training = model.training
    model.eval()
    acc_num = 0.0
    acc_den = 0.0
    try:
        for X, Y, loss_mask in val_dl:
            X = X.to(args.device, non_blocking=True)
            Y = Y.to(args.device, non_blocking=True)
            loss_mask = loss_mask.to(args.device, non_blocking=True)
            with ctx:
                with torch.cuda.amp.autocast(enabled=(args.dtype != "float32")):
                    res = model(X)
                    logits = res.logits.float()
                    lp_sum, n_tok = masked_ce_batch(logits, Y, loss_mask)
            if n_tok > 0 and not torch.isnan(lp_sum):
                acc_num += float(lp_sum.item()) * n_tok
                acc_den += n_tok
    finally:
        if was_training:
            model.train()
    if acc_den <= 0:
        return float("nan")
    return acc_num / acc_den


def run_validation(
    epoch_idx: int,
    wandb_run,
    ck_state: dict,
    val_dl: DataLoader | None,
    val_ds: SFTIntentMessagesDataset | None,
):
    """每个 epoch 结束后在验证集上计算损失；仅在 rank 0 上跑推理与写 best（DDP 时用 barrier 同步）。"""
    if val_dl is None:
        return
    if ddp:
        dist.barrier()

    rank0 = (not ddp) or dist.get_rank() == 0
    vloss: float | None = None
    vacc: float | None = None
    if rank0:
        t0 = time.time()
        vloss = compute_val_masked_ce_mean(val_dl)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        vacc, n_scored, val_rows = eval_val_predictions(val_ds, val_dl, pad_id)
        dt = time.time() - t0
        if not args.no_val_results_csv and val_ds is not None and val_rows:
            epoch_csv = os.path.join(
                args.val_results_dir,
                f"val_epoch{epoch_idx + 1:02d}.csv",
            )
            write_val_results_csv(val_rows, epoch_csv)
            n_csv_match = sum(1 for r in val_rows if r.get("matches_answer_idx"))
            Logger(
                f"验证逐条结果已写入: {epoch_csv} "
                f"(matches_answer_idx {n_csv_match}/{len(val_rows)})"
            )
        if math.isnan(vloss) and math.isnan(vacc):
            Logger(f"⚠️ Epoch [{epoch_idx + 1}] val 指标无效，跳过保存 best")
        else:
            parts = []
            if not math.isnan(vloss):
                parts.append(f"masked CE {vloss:.4f}")
            if not math.isnan(vacc):
                parts.append(f"首字母 token acc {vacc:.4f} ({n_scored} 条)")
            Logger(
                "Epoch [{}] validation {} ({} batches, {:.1f}s)".format(
                    epoch_idx + 1,
                    " | ".join(parts),
                    len(val_dl),
                    dt,
                )
            )
            if wandb_run is not None:
                log_d = {"epoch": epoch_idx + 1}
                if not math.isnan(vloss):
                    log_d["val_loss"] = vloss
                if not math.isnan(vacc):
                    log_d["val_answer_letter_acc"] = vacc
                wandb_run.log(log_d)

            use_acc = args.best_metric == "val_answer_acc" and not math.isnan(vacc)
            improved = False
            if use_acc and vacc > ck_state["best_val_answer_acc"]:
                ck_state["best_val_answer_acc"] = vacc
                improved = True
                metric_msg = f"val 首字母 acc {vacc:.4f}"
            elif (not use_acc) and (not math.isnan(vloss)) and vloss < ck_state["best_val_loss"]:
                ck_state["best_val_loss"] = vloss
                improved = True
                metric_msg = f"val masked CE {vloss:.6f}"

            if improved:
                best_path = save_checkpoint(kind="best")
                if best_path:
                    Logger(f"New best ({args.best_metric}) {metric_msg} → saved: {best_path}")
                if (
                    not args.no_val_results_csv
                    and val_ds is not None
                    and val_rows
                ):
                    best_csv = os.path.join(args.val_results_dir, "val_best.csv")
                    write_val_results_csv(val_rows, best_csv)
                    Logger(f"best 验证逐条结果已写入: {best_csv}")
                if wandb_run is not None:
                    if use_acc:
                        wandb_run.log(
                            {"val_best_answer_acc_seen": ck_state["best_val_answer_acc"]}
                        )
                    else:
                        wandb_run.log({"val_best_loss_seen": ck_state["best_val_loss"]})
            elif not math.isnan(vloss) and vloss < ck_state["best_val_loss"]:
                ck_state["best_val_loss"] = vloss

    if ddp:
        dist.barrier()


def train_epoch(epoch: int, wandb, ck_state: dict, use_train_for_best_only: bool) -> bool:
    start_time = time.time()
    if ddp and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.sft_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            with torch.cuda.amp.autocast(enabled=(args.dtype != "float32")):
                res = model(X)
                logits = res.logits.float()
                loss, n_sup = masked_ce_batch(logits, Y, loss_mask)
                loss = loss / args.accumulation_steps

        if torch.isnan(loss):
            Logger(f"⚠️ Warning: NaN loss at step {step}, skipping this step")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            step_loss = loss.item() * args.accumulation_steps
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.4f} sup_tok/batch:{} lr:{:.2e} epoch_time_est:{:.1f}min ".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    step_loss,
                    n_sup,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch / 60 - spend_time / 60,
                )
            )
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": step_loss,
                        "supervised_tokens": n_sup,
                        "lr": optimizer.param_groups[-1]["lr"],
                    }
                )

        step_loss = loss.item() * args.accumulation_steps
        if use_train_for_best_only and (not ddp or dist.get_rank() == 0) and (
            step_loss < ck_state["best_train_loss"]
        ):
            ck_state["best_train_loss"] = step_loss
            best_path = save_checkpoint(kind="best")
            if best_path:
                Logger(
                    f"New best train masked CE loss {step_loss:.6f} → saved: {best_path}"
                )
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"train_best_loss_seen": ck_state["best_train_loss"]})

        if args.early_stop_loss > 0 and step_loss < args.early_stop_loss:
            ckp = save_checkpoint(kind="latest")
            Logger(
                f"Early stop: loss={step_loss:.6f} < {args.early_stop_loss} "
                f"(epoch {epoch + 1}, step {step}), latest weights: {ckp}"
            )
            if ddp:
                dist.barrier()
            return True

        if (step + 1) % args.save_interval == 0:
            ckp = save_checkpoint(kind="latest")
            if ckp:
                Logger(f"Latest model saved to: {ckp}")

    return False


def init_model(lm_config, model_type: str, language: str, init_ckpt: str):
    tok_dir = PROJECT_ROOT / "model" / ("tokenizer_zh" if language == "zh" else "tokenizer_en")
    tokenizer = LlamaTokenizerFast.from_pretrained(str(tok_dir))

    if model_type == "QF":
        model = QubitFormer(lm_config)
    else:
        model = MiniMindForCausalLM(lm_config)

    load_path = init_ckpt.strip()
    if load_path and os.path.isfile(load_path):
        Logger(f"Loading base checkpoint (SFT or other): {load_path}")
        state_dict = torch.load(load_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            Logger(f"Missing keys (random init): {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if unexpected:
            Logger(f"Unexpected keys (ignored): {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
    else:
        if load_path:
            Logger(f"⚠️ init_ckpt 不存在: {load_path}，回退加载预训练权重")
        pretrain_ckp = os.path.join(
            args.save_dir,
            f"pretrain_{model_type}_{language}_{lm_config.dim}_final.pth",
        )
        if not os.path.exists(pretrain_ckp):
            old_ckp = os.path.join(args.save_dir, f"pretrain_{lm_config.dim}_final.pth")
            if os.path.exists(old_ckp):
                pretrain_ckp = old_ckp
        if os.path.exists(pretrain_ckp):
            Logger(f"Loading pretrained weights: {pretrain_ckp}")
            state_dict = torch.load(pretrain_ckp, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            Logger("⚠️ 未找到 init_ckpt 与 pretrain，自随机初始化开始")

    Logger(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
    )
    model = model.to(args.device)
    if args.use_torch_compile and hasattr(torch, "compile"):
        Logger("torch.compile enabled (如遇 OOM 可加 --no_torch_compile)")
        model = torch.compile(model, mode="default")
    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT on MedQA intent JSON (messages)")
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "out"))
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="已有英文 SFT 权重 .pth（推荐 parameters/sft_QF_en_1024.pth，由 parameters/QF_Struct_300M 转换）；为空则尝试加载 out_dir 下 pretrain_*",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "RAG_MedQA_US_train_10178_intent.json"
        ),
        help="含 messages 的 JSON 数组文件",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="",
        help="可选验证集 JSON（结构与训练集一致）。若提供：每个 epoch 结束在 rank0 上算 val masked CE，并以验证损失写入 *_best.pth；否则仍按训练 batch loss 写入 best",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="单步 micro-batch（A800 80GB + max_seq_len=4096 时 8 较稳，可按显存改）",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数，等效 batch ≈ batch_size * accumulation_steps * world_size",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="QubitFormer-SFT-Intent")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=2, type=int)
    parser.add_argument(
        "--max_seq_len",
        default=4096,
        type=int,
        help="意图数据 user 很长，建议 4096；显存不足可降到 2048 并略增 batch",
    )
    parser.add_argument("--sft_lr", type=float, default=5e-5, help="本阶段实际优化器学习率")
    parser.add_argument("--model_type", type=str, default="QF", choices=["QF", "MM"])
    parser.add_argument("--language", type=str, default="en", choices=["zh", "en"])
    parser.add_argument(
        "--no_torch_compile",
        action="store_true",
        help="关闭 torch.compile 以节省编译期显存或规避兼容问题",
    )
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=0,
        help="当 answer 区 masked CE（不含 pad）低于该阈值时保存并退出；默认 0 关闭",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_answer_acc",
        choices=("val_answer_acc", "val_loss"),
        help="有 --val_data_path 时按何指标保存 *_best.pth（推荐 val_answer_acc）",
    )
    parser.add_argument(
        "--val_results_dir",
        type=str,
        default="",
        help="每个 epoch 验证逐条 CSV 输出目录（列格式同 results/ablation_p3_2/intent_dev_*.csv）；"
        "默认 {out_dir}/val_results",
    )
    parser.add_argument(
        "--no_val_results_csv",
        action="store_true",
        help="不写验证逐条 CSV",
    )

    args = parser.parse_args()
    args.use_torch_compile = not args.no_torch_compile
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)
    if not args.no_val_results_csv and not (args.val_results_dir or "").strip():
        args.val_results_dir = os.path.join(args.save_dir, "val_results")

    if args.language == "zh":
        vocab_size = 6400
    else:
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

    log_path = os.path.join(args.save_dir, f"sft_intent_{args.model_type}_{args.language}_{args.hidden_size}.log")
    setup_logger(log_path)

    if args.dtype == "float32":
        ctx = nullcontext()
    elif args.dtype == "float16":
        ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
    else:
        args.device = torch.device(args.device)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"SFT-Intent-{args.model_type}-{args.language}-bs{args.batch_size}-acc{args.accumulation_steps}",
            config=vars(args),
        )
        wandb_run = wandb
    else:
        wandb_run = None

    model, tokenizer = init_model(lm_config, args.model_type, args.language, args.init_ckpt)

    data_path = args.data_path
    if not os.path.isfile(data_path):
        alt = PROJECT_ROOT / "dataset" / os.path.basename(data_path)
        if alt.is_file():
            data_path = str(alt)
            Logger(f"使用数据路径: {data_path}")
        else:
            raise SystemExit(f"--data_path 不存在: {args.data_path}")

    train_ds = SFTIntentMessagesDataset(data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    val_dl: DataLoader | None = None
    val_ds: SFTIntentMessagesDataset | None = None
    raw_val_path = (args.val_data_path or "").strip()
    if raw_val_path:
        val_path = raw_val_path
        if not os.path.isfile(val_path):
            alt = PROJECT_ROOT / "dataset" / os.path.basename(val_path)
            if alt.is_file():
                val_path = str(alt)
                Logger(f"使用验证路径: {val_path}")
            else:
                raise SystemExit(f"--val_data_path 不存在: {raw_val_path}")
        val_ds = SFTIntentMessagesDataset(val_path, tokenizer, max_length=args.max_seq_len)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
        )
        Logger(f"验证集: {val_path} | 样本数: {len(val_ds)}")
        if not args.no_val_results_csv:
            Logger(f"验证逐条 CSV 目录: {args.val_results_dir}")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.sft_lr)

    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)

    Logger("=" * 80)
    Logger("SFT Intent (messages JSON) — 二阶段微调")
    Logger(f"数据: {data_path} | 样本数: {len(train_ds)}")
    Logger(
        f"等效 batch ≈ {args.batch_size} * {args.accumulation_steps}"
        + (f" * {int(os.environ['WORLD_SIZE'])}" if ddp else "")
    )
    Logger(f"max_seq_len={args.max_seq_len} device={args.device} dtype={args.dtype} lr={args.sft_lr}")
    if args.early_stop_loss > 0:
        Logger(f"early_stop_loss={args.early_stop_loss} (masked CE，低于该值则保存并退出)")
    stem = checkpoint_stem()
    if raw_val_path:
        Logger(
            f"checkpoint: `{stem}.pth`=最新; `{stem}_best.pth`=验证集 **{args.best_metric}** 最优 "
            "(val 仅在 rank 0；日志含 masked CE 与首字母 token acc)"
        )
    else:
        Logger(
            f"checkpoint: `{stem}.pth`=最新; `{stem}_best.pth`=**训练 batch** masked CE 历史最低 "
            "(未设置 --val_data_path)"
        )
    Logger("=" * 80)

    early_stopped = False
    ck_state = {
        "best_train_loss": float("inf"),
        "best_val_loss": float("inf"),
        "best_val_answer_acc": -1.0,
    }
    use_train_for_best_only = val_dl is None

    if (not ddp) or dist.get_rank() == 0:
        X0, Y0, m0 = train_ds[0]
        Logger(
            f"单条样本监督 token 数（非 pad）: {int(m0.sum())} "
            f"（单字母标签通常 2–3：选项 + </s>；此前若含 pad 会虚增至数千）"
        )
    for epoch in range(args.epochs):
        Logger(f"\nEpoch {epoch + 1}/{args.epochs}")
        if train_epoch(epoch, wandb_run, ck_state, use_train_for_best_only):
            early_stopped = True
            break
        run_validation(epoch, wandb_run, ck_state, val_dl, val_ds)

    if not early_stopped and (not ddp or dist.get_rank() == 0):
        ckp = save_checkpoint(kind="latest")
        best_parts = []
        if ck_state["best_val_answer_acc"] >= 0:
            best_parts.append(
                f"best **val** 首字母 acc: {ck_state['best_val_answer_acc']:.4f} → `{stem}_best.pth`"
            )
        elif ck_state["best_val_loss"] < float("inf"):
            best_parts.append(
                f"best **val** masked CE: {ck_state['best_val_loss']:.6f} → `{stem}_best.pth`"
            )
        elif ck_state["best_train_loss"] < float("inf"):
            best_parts.append(
                f"best **train batch** masked CE: {ck_state['best_train_loss']:.6f} → `{stem}_best.pth`"
            )
        best_note = " | " + "; ".join(best_parts) if best_parts else ""
        Logger(f"Final latest weights: {ckp}{best_note}")

    if wandb_run is not None:
        wandb_run.finish()