import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import glob
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
import gc
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast
from model.QFConfig import QFConfig

from model.model import MiniMindForCausalLM
from model.qbmodel import QubitFormer
from dataset.lm_zh_dataset import PretrainDataset as PretrainDatasetZh
from dataset.lm_en_dataset import PretrainDataset as PretrainDatasetEn
import numpy as np
from collections import deque

os.environ["ASCEND_GE_USE_STATIC_MEMORY"] = "1"
os.environ["ASCEND_GE_MEMORY_CACHE_SIZE"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ThroughputMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.step_times = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        self.start_time = None
        self.total_tokens = 0
        self.total_steps = 0

    def start_batch(self):
        self.batch_start = time.time()

    def end_batch(self, batch_size, seq_length, tokens_per_sample=None):
        batch_time = time.time() - self.batch_start
        if tokens_per_sample is not None:
            tokens = tokens_per_sample
        else:
            tokens = batch_size * seq_length
        self.step_times.append(batch_time)
        self.token_counts.append(tokens)
        self.total_tokens += tokens
        self.total_steps += 1
        return tokens, batch_time

    def get_throughput(self):
        if len(self.step_times) == 0:
            return 0, 0, 0
        avg_time = np.mean(self.step_times)
        avg_tokens = np.mean(self.token_counts)
        tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        samples_per_sec = len(self.step_times) / np.sum(self.step_times)
        batches_per_sec = 1.0 / avg_time if avg_time > 0 else 0
        return tokens_per_sec, samples_per_sec, batches_per_sec

    def get_avg_batch_time(self):
        return np.mean(self.step_times) if len(self.step_times) > 0 else 0

    def reset(self):
        self.step_times.clear()
        self.token_counts.clear()
        self.total_tokens = 0
        self.total_steps = 0


def get_device_info(device):
    device_info = {}
    if torch.npu.is_available() and "npu" in str(device):
        device_info["type"] = "NPU"
        try:
            device_info["name"] = torch.npu.get_device_name(device)
            device_info["capability"] = torch.npu.get_device_capability(device)
            device_info["memory_total"] = (
                torch.npu.get_device_properties(device).total_memory / 1024**3
            )
        except:
            device_info["name"] = "Unknown NPU"
            device_info["memory_total"] = 0
    elif torch.cuda.is_available() and "cuda" in str(device):
        device_info["type"] = "CUDA"
        device_info["name"] = torch.cuda.get_device_name(device)
        device_info["capability"] = torch.cuda.get_device_capability(device)
        device_info["memory_total"] = (
            torch.cuda.get_device_properties(device).total_memory / 1024**3
        )
    else:
        device_info["type"] = "CPU"
        device_info["name"] = platform.processor()
        device_info["memory_total"] = 0
    return device_info


def Logger(content, rank=0):
    if not ddp or dist.get_rank() == rank:
        print(f"[Rank {dist.get_rank() if ddp else 0}] {content}")


def get_lr(current_update_step, total_update_steps, lr):
    warmup_steps = int(0.01 * total_update_steps)
    max_lr = lr
    if current_update_step < warmup_steps:
        return max_lr * (current_update_step / warmup_steps)
    decay_ratio = (current_update_step - warmup_steps) / (
        total_update_steps - warmup_steps
    )
    return max_lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))


def cleanup_memory():
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()
        allocated = torch.npu.memory_allocated() / 1024**3
        reserved = torch.npu.memory_reserved() / 1024**3
        Logger(f"NPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        Logger(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def cleanup_old_checkpoints(save_dir, keep=3):
    checkpoints = sorted(glob.glob(f"{save_dir}/*.pth"), key=os.path.getmtime)
    for ckpt in checkpoints[:-keep]:
        os.remove(ckpt)
        Logger(f"🗑️ Removed old checkpoint: {ckp}")


def train_epoch(
    epoch,
    wandb,
    model,
    train_loader,
    optimizer,
    scaler,
    ctx,
    args,
    lm_config,
    iter_per_epoch,
    ddp,
):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    total_loss = 0
    log_steps = 0
    throughput_monitor = ThroughputMonitor(window_size=args.log_interval)
    epoch_start_time = time.time()

    cleanup_memory()
    model.train()

    # 用于学习率调度的更新步数计数
    update_step = 0
    total_update_steps = (iter_per_epoch // args.accumulation_steps) * args.epochs
    # 初始化当前学习率（从优化器中获取）
    current_lr = optimizer.param_groups[0]["lr"]

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        try:
            throughput_monitor.start_batch()
            X = X.to(args.device, non_blocking=True)
            Y = Y.to(args.device, non_blocking=True)
            loss_mask = loss_mask.to(args.device, non_blocking=True)

            with ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)), Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()
            log_steps += 1

            if (step + 1) % args.accumulation_steps == 0:
                # 更新学习率
                lr = get_lr(update_step, total_update_steps, args.learning_rate)
                current_lr = lr  # 保存到外层变量
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            actual_tokens = loss_mask.sum().item()
            tokens, batch_time = throughput_monitor.end_batch(
                args.batch_size, lm_config.max_seq_len, tokens_per_sample=actual_tokens
            )

            if step % args.log_interval == 0 and log_steps > 0:
                avg_loss = total_loss / log_steps * args.accumulation_steps
                spend_time = time.time() - start_time
                estimated_total_time = spend_time / (step + 1) * iter_per_epoch
                remaining_time = estimated_total_time - spend_time

                tokens_per_sec, samples_per_sec, batches_per_sec = (
                    throughput_monitor.get_throughput()
                )
                avg_batch_time = throughput_monitor.get_avg_batch_time()

                # MFU估算（简化）
                if tokens_per_sec > 0:
                    params = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    flops_per_token = 6 * params
                    total_flops_per_sec = tokens_per_sec * flops_per_token
                    if torch.npu.is_available():
                        peak_flops = 2.5e15  # NPU 910B
                    elif torch.cuda.is_available():
                        peak_flops = 312e12  # A100
                    else:
                        peak_flops = 1e12
                    mfu_estimate = total_flops_per_sec / peak_flops
                else:
                    mfu_estimate = 0

                Logger(
                    "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.8f} elapsed:{:.1f}min remaining:{:.1f}min".format(
                        epoch + 1,
                        args.epochs,
                        step,
                        iter_per_epoch,
                        avg_loss,
                        current_lr,
                        spend_time / 60,
                        remaining_time / 60,
                    )
                )
                Logger(
                    "🚀 Throughput: {:.1f} tok/s, {:.1f} samples/s, {:.2f} ms/batch, MFU: {:.1%}".format(
                        tokens_per_sec,
                        samples_per_sec,
                        avg_batch_time * 1000,
                        mfu_estimate,
                    )
                )

                if wandb is not None and (not ddp or dist.get_rank() == 0):
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "lr": current_lr,
                            "epoch": epoch + 1,
                            "step": step,
                            "update_step": update_step,
                            "elapsed_time": spend_time / 60,
                            "memory_gb": (
                                torch.npu.memory_allocated() / 1024**3
                                if torch.npu.is_available()
                                else 0
                            ),
                            "throughput_tokens_per_sec": tokens_per_sec,
                            "throughput_samples_per_sec": samples_per_sec,
                            "batch_time_ms": avg_batch_time * 1000,
                            "mfu_estimate": mfu_estimate,
                            "actual_tokens_per_batch": actual_tokens,
                            "total_tokens_processed": throughput_monitor.total_tokens,
                        }
                    )

                total_loss = 0
                log_steps = 0

            if (step + 1) % args.save_interval == 0 and (
                not ddp or dist.get_rank() == 0
            ):
                model.eval()
                ckp = f"{args.save_dir}/pretrain_{args.model_type}_{args.language}_{lm_config.dim}.pth"
                raw_model = (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                )
                raw_model = (
                    raw_model._orig_mod
                    if hasattr(raw_model, "_orig_mod")
                    else raw_model
                )
                state_dict = raw_model.state_dict()
                if args.save_dtype == "float16":
                    state_dict = {k: v.half() for k, v in state_dict.items()}
                elif args.save_dtype == "bfloat16":
                    state_dict = {k: v.bfloat16() for k, v in state_dict.items()}
                save_dict = {
                    "state_dict": state_dict,
                    "config": lm_config,
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "update_step": update_step,
                }
                torch.save(save_dict, ckp)
                Logger(f"Saved checkpoint: {ckp}")
                cleanup_old_checkpoints(args.save_dir, keep=3)
                model.train()

        except Exception as e:
            Logger(f"Error at epoch {epoch}, step {step}: {str(e)}")
            cleanup_memory()
            raise

    epoch_time = time.time() - epoch_start_time
    avg_tokens_per_sec = (
        throughput_monitor.total_tokens / epoch_time if epoch_time > 0 else 0
    )
    Logger(
        f"📊 Epoch {epoch+1} summary: Time: {epoch_time:.1f}s ({epoch_time/60:.1f}min), "
        f"Avg throughput: {avg_tokens_per_sec:.1f} tokens/s, Total tokens: {throughput_monitor.total_tokens:,}"
    )
    cleanup_memory()
    return {
        "epoch": epoch + 1,
        "total_tokens": throughput_monitor.total_tokens,
        "epoch_time": epoch_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
    }


def init_model(lm_config, args):
    """Initialize tokenizer and model based on model_type and language."""
    Logger(f"Initializing tokenizer for language: {args.language}")
    if args.language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("../model/tokenizer_zh")
    else:  # en
        tokenizer = LlamaTokenizerFast.from_pretrained("../model/tokenizer_en")

    Logger(f"Initializing model type: {args.model_type}")
    if args.model_type == "QF":
        model = QubitFormer(lm_config)
    else:  # MM
        model = MiniMindForCausalLM(lm_config)

    if "npu" in str(args.device):
        try:
            import torch_npu

            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.config.allow_internal_format = True
            torch.npu.config.allow_fuse_layernorm = True
            Logger("NPU optimizations enabled")
        except ImportError:
            Logger("torch_npu not available, using standard training")

    model = model.to(args.device)
    # 启用梯度检查点（减少显存）
    # model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"Total trainable parameters: {total_params / 1e6:.2f} M")
    return model, tokenizer, total_params


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    try:
        dist.init_process_group(backend="hccl")
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        DEVICE = f"npu:{ddp_local_rank}"
        torch.npu.set_device(DEVICE)
        Logger(
            f"Distributed init: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}"
        )
    except Exception as e:
        Logger(f"Failed to init distributed mode: {e}")
        raise


def print_training_summary(args, lm_config, model_params, device_info, total_stats):
    Logger("=" * 80)
    Logger("🎯 Training Summary")
    Logger("=" * 80)
    Logger(f"Hardware: {device_info['type']} - {device_info['name']}")
    if device_info["memory_total"] > 0:
        Logger(f"Memory: {device_info['memory_total']:.1f} GB")
    Logger(f"Model type: {args.model_type}")
    Logger(f"Language: {args.language}")
    Logger(f"Model parameters: {model_params / 1e6:.1f} M")
    Logger(f"Hidden size: {lm_config.dim}")
    Logger(f"Number of layers: {lm_config.n_layers}")
    Logger(f"Sequence length: {lm_config.max_seq_len}")
    Logger(f"Vocabulary size: {lm_config.vocab_size}")
    Logger(f"Batch size: {args.batch_size}")
    Logger(f"Gradient accumulation: {args.accumulation_steps}")
    world_size = dist.get_world_size() if ddp else 1
    Logger(f"World size: {world_size}")
    Logger(
        f"Effective batch size: {args.batch_size * args.accumulation_steps * world_size}"
    )
    Logger(f"Learning rate: {args.learning_rate}")
    Logger(f"Epochs: {args.epochs}")
    if total_stats:
        total_tokens = sum(stat["total_tokens"] for stat in total_stats)
        total_time = sum(stat["epoch_time"] for stat in total_stats)
        avg_throughput = total_tokens / total_time if total_time > 0 else 0
        Logger(f"Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
        Logger(f"Total tokens processed: {total_tokens:,}")
        Logger(f"Average throughput: {avg_throughput:.1f} tokens/s")
        Logger(f"Total samples: {total_tokens / lm_config.max_seq_len:,.0f}")
        if device_info["type"] == "CUDA":
            cost_per_hour = 3.0
            training_hours = total_time / 3600
            estimated_cost = training_hours * cost_per_hour
            Logger(
                f"Estimated training cost: ${estimated_cost:.2f} (based on A100 ${cost_per_hour}/hour)"
            )
    Logger("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QubitFormer/MiniMind Pretraining on NPU"
    )
    parser.add_argument("--out_dir", type=str, default="out_qf10")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="npu:0" if torch.npu.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PE-Experiment")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="QF",
        choices=["QF", "MM"],
        help="Model type: QF (QubitFormer) or MM (MiniMind)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language: zh or en",
    )

    # Model architecture parameters
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--num_hidden_layers", default=24, type=int)
    parser.add_argument("--num_attention_heads", default=16, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--max_position_embeddings", default=1024, type=int)
    parser.add_argument(
        "--vocab_size",
        default=None,
        type=int,
        help="Vocabulary size (auto-set based on language if not provided)",
    )
    parser.add_argument("--dropout", default=0.0, type=float)

    # Training parameters
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="")

    args = parser.parse_args()

    # Set vocabulary size based on language if not explicitly provided
    if args.vocab_size is None:
        if args.language == "zh":
            args.vocab_size = 6400
        else:  # en
            args.vocab_size = 32000
        Logger(f"Auto-set vocab_size to {args.vocab_size} for language {args.language}")

    # Create model configuration
    lm_config = QFConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_position_embeddings,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        flash_attn=True,  # Enable Flash Attention on NPU
        multiple_of=64,
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(1337)
    if torch.npu.is_available():
        torch.npu.manual_seed(1337)

    device_str = str(args.device)
    device_type = (
        "npu" if "npu" in device_str else ("cuda" if "cuda" in device_str else "cpu")
    )

    ddp = int(os.environ.get("RANK", -1)) != -1 or args.ddp
    ddp_local_rank, DEVICE = 0, args.device

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    else:
        args.device = torch.device(args.device)

    device_info = get_device_info(args.device)
    model, tokenizer, model_params = init_model(lm_config, args)

    Logger(f"🔧 Training device: {device_info['type']}: {device_info['name']}")
    if device_info["memory_total"] > 0:
        Logger(f"🔧 Device memory: {device_info['memory_total']:.1f} GB")
    Logger("📋 Training configuration:")
    Logger(f"  Model type: {args.model_type}")
    Logger(f"  Language: {args.language}")
    Logger(f"  Batch size: {args.batch_size}")
    Logger(f"  Gradient accumulation steps: {args.accumulation_steps}")
    world_size = dist.get_world_size() if ddp else 1
    Logger(f"  World size: {world_size}")
    Logger(
        f"  Effective batch size: {args.batch_size * args.accumulation_steps * world_size}"
    )
    Logger(f"  Sequence length: {lm_config.max_seq_len}")
    Logger(f"  Model parameters: {model_params / 1e6:.1f}M")

    if device_type == "cpu":
        ctx = nullcontext()
        scaler = None
    else:
        if "npu" in device_str:
            if not torch.npu.is_initialized():
                torch.npu.init()
            ctx = torch.npu.amp.autocast(
                enabled=(args.dtype in ["float16", "bfloat16"])
            )
            scaler = torch.npu.amp.GradScaler(
                enabled=(args.dtype in ["float16", "bfloat16"]), init_scale=8192.0
            )
        else:
            ctx = torch.cuda.amp.autocast(
                enabled=(args.dtype in ["float16", "bfloat16"])
            )
            scaler = torch.cuda.amp.GradScaler(
                enabled=(args.dtype in ["float16", "bfloat16"])
            )

    wandb = None
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        try:
            import wandb

            if not args.wandb_run_name:
                args.wandb_run_name = f"{args.model_type}_{args.language}_{args.hidden_size}-l{args.num_hidden_layers}-lr{args.learning_rate}"
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    **vars(args),
                    "model_params": model_params,
                    "device_info": device_info,
                },
            )
            Logger(f"📊 wandb initialized: {args.wandb_project}/{args.wandb_run_name}")
        except ImportError:
            Logger("⚠️ wandb not available, using None")

    if args.resume or args.checkpoint_path:
        ckp = (
            args.checkpoint_path
            if args.checkpoint_path
            else f"{args.save_dir}/pretrain_{args.model_type}_{args.language}_{lm_config.dim}.pth"
        )
        if os.path.exists(ckp):
            Logger(f"🔍 Loading checkpoint from {ckp}")
            state_dict = torch.load(ckp, map_location=args.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "") if "_orig_mod." in k else k
                new_state_dict[new_key] = v
            if isinstance(model, DistributedDataParallel):
                model.module.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict, strict=False)
            Logger(f"✅ Resumed training from checkpoint {ckp}")
        else:
            Logger(f"❌ Checkpoint {ckp} not found, starting from scratch")

    Logger("📂 Loading dataset...")
    # Choose dataset class based on language
    if args.language == "zh":
        train_ds = PretrainDatasetZh(
            args.data_path, tokenizer, max_length=lm_config.max_seq_len
        )
    else:
        train_ds = PretrainDatasetEn(
            args.data_path, tokenizer, max_length=lm_config.max_seq_len
        )
    Logger(f"📊 Dataset loaded: {len(train_ds)} samples")

    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.1,
    )

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {
            "cos_table",
            "sin_table",
        }  # Ignore RoPE buffers
        model = DistributedDataParallel(
            model, device_ids=[ddp_local_rank], find_unused_parameters=False
        )

    iter_per_epoch = len(train_loader)
    Logger(
        f"⚡ Training steps per epoch: {iter_per_epoch}, total epochs: {args.epochs}"
    )

    training_stats = []

    try:
        for epoch in range(args.epochs):
            if ddp:
                train_sampler.set_epoch(epoch)
            cleanup_memory()
            Logger(f"🚀 Starting epoch {epoch + 1}/{args.epochs}")
            epoch_stats = train_epoch(
                epoch,
                wandb,
                model,
                train_loader,
                optimizer,
                scaler,
                ctx,
                args,
                lm_config,
                iter_per_epoch,
                ddp,
            )
            training_stats.append(epoch_stats)
            cleanup_memory()
            if torch.npu.is_available():
                Logger(
                    f"📈 Epoch {epoch+1} finished, NPU memory: {torch.npu.memory_allocated()/1024**3:.2f} GB"
                )
            elif torch.cuda.is_available():
                Logger(
                    f"📈 Epoch {epoch+1} finished, GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                )

        if not ddp or ddp_local_rank == 0:
            model.eval()
            ckp = (
                f"{args.save_dir}/pretrain_{args.model_type}_{args.language}_final.pth"
            )
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = (
                raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
            )
            state_dict = raw_model.state_dict()
            if args.save_dtype == "float16":
                state_dict = {k: v.half() for k, v in state_dict.items()}
            elif args.save_dtype == "bfloat16":
                state_dict = {k: v.bfloat16() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            Logger(f"✅ Final model saved to {ckp}")

        print_training_summary(
            args, lm_config, model_params, device_info, training_stats
        )
        Logger("🎉 Training completed successfully!")

    except Exception as e:
        Logger(f"❌ Training failed with error: {str(e)}")
        import traceback

        Logger(traceback.format_exc())
        cleanup_memory()
        if training_stats:
            print_training_summary(
                args, lm_config, model_params, device_info, training_stats
            )
        raise
