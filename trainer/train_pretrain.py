import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast
from model.QFConfig import QFConfig

from model.model import MiniMindForCausalLM
from model.qbmodel import QubitFormer
from dataset.lm_zh_dataset import PretrainDataset as PretrainDatasetZh
from dataset.lm_en_dataset import PretrainDataset as PretrainDatasetEn
import torch._inductor.config as cfg
import logging

os.environ["TORCHINDUCTOR_CUDAGRAPH"] = "0"
warnings.filterwarnings("ignore")

# Global logger
logger = None


def setup_logger(log_file):
    """Setup logger to output to both console and file"""
    global logger
    logger = logging.getLogger("MiniMindPretrain")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def Logger(content):
    """Unified logging interface"""
    if not ddp or dist.get_rank() == 0:
        if logger:
            logger.info(content)
        else:
            print(content)


def get_lr(current_step, total_steps, lr):
    """Cosine annealing learning rate scheduler"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def format_time(seconds):
    """Format seconds into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class GPUTimer:
    """Measure GPU computation time accurately using CUDA events"""

    def __init__(self, device=None):
        self.device = (
            device or torch.cuda.current_device() if torch.cuda.is_available() else None
        )
        self.start_event = None
        self.end_event = None
        self.elapsed_time_ms = 0.0

    def start(self):
        """Call before GPU computation starts"""
        if self.device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

    def end(self):
        """Call after GPU computation ends"""
        if (
            self.device is not None
            and torch.cuda.is_available()
            and self.start_event is not None
        ):
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

    def get_seconds(self):
        """Return seconds"""
        return self.elapsed_time_ms / 1000.0

    def get_milliseconds(self):
        """Return milliseconds"""
        return self.elapsed_time_ms

    def reset(self):
        """Reset timer"""
        self.elapsed_time_ms = 0.0


class GPUMetrics:
    """GPU performance metrics collector"""

    def __init__(self, warmup_steps=20):
        self.warmup_steps = warmup_steps
        self.gpu_times = []
        self.throughputs = []
        self.current_step = 0

    def record_step(self, gpu_time_seconds, tokens_processed):
        """Record GPU metrics for one step"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            return None

        gpu_throughput = (
            tokens_processed / gpu_time_seconds if gpu_time_seconds > 0 else 0
        )

        self.gpu_times.append(gpu_time_seconds)
        self.throughputs.append(gpu_throughput)

        return gpu_throughput

    def get_recent_avg(self, window_size=50):
        """Get average of recent N steps"""
        if len(self.gpu_times) == 0:
            return 0.0, 0.0

        recent_steps = min(window_size, len(self.gpu_times))
        avg_gpu_time = sum(self.gpu_times[-recent_steps:]) / recent_steps
        avg_throughput = sum(self.throughputs[-recent_steps:]) / recent_steps

        return avg_gpu_time, avg_throughput

    def get_overall_avg(self):
        """Get overall average (excluding warmup)"""
        if len(self.gpu_times) == 0:
            return 0.0, 0.0

        avg_gpu_time = sum(self.gpu_times) / len(self.gpu_times)
        avg_throughput = sum(self.throughputs) / len(self.throughputs)

        return avg_gpu_time, avg_throughput

    def reset(self):
        """Reset metrics"""
        self.gpu_times = []
        self.throughputs = []
        self.current_step = 0


def train_epoch(epoch, wandb, global_step_start, total_start_time):
    """Train one epoch"""
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    epoch_start_time = time.time()

    gpu_timer = GPUTimer(args.device)
    gpu_metrics = GPUMetrics(warmup_steps=20)

    world_size = dist.get_world_size() if ddp else 1

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        data_load_start = time.time()
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)
        data_load_time = time.time() - data_load_start

        lr = get_lr(
            global_step_start + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        gpu_timer.start()

        with ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        gpu_timer.end()

        tokens_per_step = args.batch_size * args.max_seq_len * world_size
        gpu_time_seconds = gpu_timer.get_seconds()
        gpu_throughput = gpu_metrics.record_step(gpu_time_seconds, tokens_per_step)

        if step % args.log_interval == 0:
            avg_gpu_time, avg_gpu_throughput = gpu_metrics.get_recent_avg(
                window_size=50
            )

            step_end_time = time.time()
            epoch_elapsed = step_end_time - epoch_start_time
            total_elapsed = step_end_time - total_start_time

            total_tokens_processed = tokens_per_step * args.log_interval
            end_to_end_throughput = (
                total_tokens_processed / epoch_elapsed if epoch_elapsed > 0 else 0
            )

            total_steps_done = global_step_start + step
            total_tokens_processed_sofar = total_steps_done * tokens_per_step
            avg_end_to_end_throughput = (
                total_tokens_processed_sofar / total_elapsed if total_elapsed > 0 else 0
            )

            steps_remaining = args.epochs * iter_per_epoch - total_steps_done
            eta_seconds = steps_remaining * (total_elapsed / max(total_steps_done, 1))

            gpu_efficiency = (
                avg_gpu_time / (avg_gpu_time + data_load_time)
                if (avg_gpu_time + data_load_time) > 0
                else 0
            )

            Logger(
                f"Epoch: [{epoch+1}/{args.epochs}] | "
                f"Step: [{step}/{iter_per_epoch}] | "
                f"Global Step: {global_step_start + step} | "
                f"Loss: {loss.item() * args.accumulation_steps:.3f} | "
                f"LR: {optimizer.param_groups[-1]['lr']:.2e} | "
                f"GPU Time: {avg_gpu_time*1000:.1f}ms | "
                f"Data Load Time: {data_load_time*1000:.1f}ms | "
                f"GPU Throughput: {avg_gpu_throughput:,.0f} tokens/s | "
                f"E2E Throughput: {end_to_end_throughput:,.0f} tokens/s | "
                f"GPU Efficiency: {gpu_efficiency:.1%} | "
                f"ETA: {format_time(eta_seconds)}"
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "gpu_time_ms": avg_gpu_time * 1000,
                        "data_load_time_ms": data_load_time * 1000,
                        "gpu_throughput": avg_gpu_throughput,
                        "end_to_end_throughput": end_to_end_throughput,
                        "gpu_efficiency": gpu_efficiency,
                        "epoch_time": epoch_elapsed / 60,
                        "total_time": total_elapsed,
                    },
                    step=global_step_start + step,
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()

            weight_filename = (
                f"pretrain_{args.model_type}_{args.language}_{args.hidden_size}.pth"
            )
            ckp = os.path.join(args.save_dir, weight_filename)

            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = (
                raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
            )

            state_dict = raw_model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()

            Logger(f"Model saved to: {ckp}")

    return global_step_start + len(train_loader)


def init_model(lm_config, model_type, language):
    """Initialize model and tokenizer"""
    # Choose tokenizer based on language
    if language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("../model/tokenizer_zh")
    else:  # en
        tokenizer = LlamaTokenizerFast.from_pretrained("../model/tokenizer_en")

    # Choose model based on model_type
    if model_type == "QF":
        model = QubitFormer(lm_config).to(args.device)
    else:  # MM
        model = MiniMindForCausalLM(lm_config).to(args.device)

    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")

    Logger(
        f"LLM trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
    )
    return model, tokenizer


def init_distributed_mode():
    """Initialize distributed training environment"""
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--model_type", type=str, default="QF", help="QF or MM")
    parser.add_argument("--language", type=str, default="zh", help="zh or en")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--local_rank", type=int, default=-1)

    # Model architecture parameters
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=2, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)

    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument(
        "--data_path", type=str, default="../dataset/pretrain_hq_zh.jsonl"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint path to resume training",
    )
    args = parser.parse_args()

    # Set vocab size based on language
    if args.language == "zh":
        vocab_size = 6400
    else:  # en
        vocab_size = 32000

    # Update config with dynamic vocab size
    lm_config = QFConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
        vocab_size=vocab_size,
        n_kv_heads=args.num_key_value_heads,
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    world_size = 1
    if args.ddp:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    tokens_per_iter = args.batch_size * args.max_seq_len * world_size

    device_type = "cuda" if "cuda" in args.device else "cpu"

    log_filename = f"pretrain_{args.hidden_size}.log"
    log_path = os.path.join(args.save_dir, log_filename)

    setup_logger(log_path)

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "hidden_size": args.hidden_size,
                "world_size": world_size,
                "vocab_size": vocab_size,
                "model_type": args.model_type,
                "language": args.language,
            },
        )
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, args.model_type, args.language)

    # Load checkpoint if specified
    if args.resume_from is not None and (not ddp or dist.get_rank() == 0):
        if os.path.exists(args.resume_from):
            Logger(f"Loading checkpoint from: {args.resume_from}")

            checkpoint = torch.load(args.resume_from, map_location="cpu")

            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = (
                raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
            )

            load_result = raw_model.load_state_dict(checkpoint, strict=False)

            if load_result.missing_keys:
                Logger(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                Logger(f"Unexpected keys: {load_result.unexpected_keys}")

            Logger("Checkpoint loaded successfully!")
        else:
            Logger(f"Checkpoint not found: {args.resume_from}")

    # Choose dataset class based on language
    if args.language == "zh":
        PretrainDataset = PretrainDatasetZh
    else:
        PretrainDataset = PretrainDatasetEn

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    total_steps = args.epochs * iter_per_epoch
    total_tokens = total_steps * tokens_per_iter

    Logger("=" * 80)
    Logger(f"Starting training... ")
    Logger(f"Log file: {log_path}")
    Logger(f"Training configuration:")
    Logger(
        f"  - Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
    )
    Logger(f"  - Batch size: {args.batch_size} (per GPU)")
    Logger(f"  - Sequence length: {args.max_seq_len}")
    Logger(f"  - World size: {world_size}")
    Logger(f"  - Total iterations: {total_steps}")
    Logger(f"  - Tokens per iteration: {tokens_per_iter:,}")
    Logger(f"  - Total tokens to process: {total_tokens:,}")
    Logger(f"  - Model type: {args.model_type}")
    Logger(f"  - Language: {args.language}")
    Logger(f"  - Vocabulary size: {vocab_size}")
    Logger("=" * 80)

    total_start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)   
        Logger(f"\n{'='*60}")
        Logger(f"Starting epoch {epoch+1}/{args.epochs}")
        Logger(f"{'='*60}")

        epoch_start_time = time.time()
        global_step = train_epoch(epoch, wandb, global_step, total_start_time)

        epoch_time = time.time() - epoch_start_time
        Logger(f"Epoch {epoch+1} completed in {format_time(epoch_time)}")

    total_time = time.time() - total_start_time
    Logger("\n" + "=" * 80)
    Logger("Training completed!")
    Logger(f"Total training time: {format_time(total_time)}")

    gpumeterics = GPUMetrics()
    overall_avg_gpu_time, overall_avg_gpu_throughput = gpumeterics.get_overall_avg()
    Logger(f"Average GPU time: {overall_avg_gpu_time*1000:.1f}ms")
    Logger(f"Average GPU throughput: {overall_avg_gpu_throughput:,.0f} tokens/s")
    Logger(f"End-to-end average throughput: {total_tokens/total_time:,.0f} tokens/s")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    estimated_flops_per_token = 2 * total_params
    estimated_tflops = (overall_avg_gpu_throughput * estimated_flops_per_token) / 1e12
    Logger(f"Estimated TFLOPS: {estimated_tflops:.2f} TFLOPs")

    Logger(f"Total tokens processed: {total_tokens:,}")
    Logger("=" * 80)

    # Save final model
    if not ddp or dist.get_rank() == 0:
        model.eval()

        weight_filename = (
            f"pretrain_{args.model_type}_{args.language}_{args.hidden_size}_final.pth"
        )
        ckp = os.path.join(args.save_dir, weight_filename)

        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        raw_model = (
            raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        )

        state_dict = raw_model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)
        Logger(f"Final model saved to: {ckp}")
