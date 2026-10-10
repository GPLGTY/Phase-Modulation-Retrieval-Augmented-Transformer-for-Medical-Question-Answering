import os
import platform
import argparse
import time
import math
import warnings
import logging

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from transformers import LlamaTokenizerFast, AutoTokenizer
from model.QFConfig import QFConfig
from model.model import MiniMindForCausalLM
from model.qbmodel import QubitFormer

warnings.filterwarnings("ignore")

# Global logger / 全局日志记录器
logger = None


def setup_logger(log_file):
    """Set up logger to output to both console and file
    设置日志记录器，同时输出到控制台和文件"""
    global logger
    logger = logging.getLogger("MiniMindSFT_NPU")
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
    """Unified log output interface (only rank 0 in DDP)
    统一的日志输出接口（DDP模式下仅主进程输出）"""
    if not ddp or dist.get_rank() == 0:
        if logger:
            logger.info(content)
        else:
            print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            logits = res.logits.float() if hasattr(res, "logits") else res.float()
            loss = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_time:{:.1f}min ".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch / 60 - spend_time / 60,
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_time": spend_time / (step + 1) * iter_per_epoch / 60
                        - spend_time / 60,
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            # Save filename includes model type and language / 保存文件名包含模型类型和语言
            weight_filename = (
                f"sft_{args.model_type}_{args.language}_{lm_config.dim}.pth"
            )
            ckp = os.path.join(args.save_dir, weight_filename)

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # Convert to half precision to save space / 转换为半精度以节省空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()
            Logger(f"Model saved to: {ckp}")


def init_model(lm_config, model_type, language):
    """Initialize model and tokenizer based on model type and language
    根据模型类型和语言初始化模型与分词器"""
    # 1. Load tokenizer based on language / 根据语言加载分词器
    if language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("../model/tokenizer_zh")
    else:  # en
        tokenizer = LlamaTokenizerFast.from_pretrained("../model/tokenizer_en")

    # 2. Instantiate model by model type / 按模型类型实例化模型
    if model_type == "QF":
        model = QubitFormer(lm_config)
    else:  # MM
        model = MiniMindForCausalLM(lm_config)

    # 3. Load pretrained weights (filename includes model type, language and hidden size)
    # 加载预训练权重（文件名包含模型类型、语言和隐藏维度）
    pretrain_ckp = os.path.join(
        args.save_dir,
        f"pretrain_{model_type}_{language}_{lm_config.dim}_final.pth",
    )
    # Fallback to old filename if new one does not exist / 兼容旧格式
    if not os.path.exists(pretrain_ckp):
        old_ckp = os.path.join(args.save_dir, f"pretrain_{lm_config.dim}_final.pth")
        if os.path.exists(old_ckp):
            pretrain_ckp = old_ckp
            Logger(f"Using old format pretrained weights: {pretrain_ckp}")
        else:
            Logger(f"Warning: Pretrained weights not found at {pretrain_ckp}")
            Logger("Will train SFT with random initialization!")

    if os.path.exists(pretrain_ckp):
        Logger(f"Loading pretrained weights: {pretrain_ckp}")
        state_dict = torch.load(pretrain_ckp, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            Logger(f"Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            Logger(f"Unexpected keys (ignored): {unexpected_keys}")

    Logger(
        f"Total trainable LLM parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} million"
    )
    Logger(f"Model type: {model_type.upper()} | Language: {language.upper()}")

    model = model.to(args.device)

    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")

    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="hccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"npu:{ddp_local_rank}"
    torch.npu.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT on NPU")
    parser.add_argument("--out_dir", type=str, default="parameters")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="npu:0" if torch.npu.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    # Model architecture parameters / 模型结构参数
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--num_hidden_layers", default=24, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=2, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument(
        "--data_path", type=str, default="./dataset/OpenHermes-2.5/openhermes2_5.jsonl"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    # New arguments for model type and language / 新增模型类型和语言参数
    parser.add_argument(
        "--model_type",
        type=str,
        default="MM",
        choices=["QF", "MM"],
        help="Model type: QF (QubitFormer) or MM (MiniMindForCausalLM)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language: zh or en",
    )
    parser.add_argument(
        "--sft_lr",
        type=float,
        default=1e-4,
        help="SFT specific learning rate (usually lower than pre-training)",
    )

    args = parser.parse_args()

    # Set vocabulary size based on language / 根据语言设置词汇表大小
    if args.language == "zh":
        vocab_size = 6400
    else:  # en
        vocab_size = 32000

    # Create model configuration using QFConfig (includes vocab_size) / 使用QFConfig创建模型配置（包含词汇表大小）
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
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "npu" if "npu" in args.device else "cpu"

    # Log filename includes model type and language / 日志文件名包含模型类型和语言
    log_filename = f"full_sft_{args.model_type}_{args.language}_{args.hidden_size}.log"
    log_path = os.path.join(args.save_dir, log_filename)
    setup_logger(log_path)

    args.wandb_run_name = f"MiniMind-Full-SFT-{args.model_type}-{args.language}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"

    # Mixed precision context / 混合精度上下文
    if args.dtype == "float32":
        ctx = nullcontext()
    elif args.dtype == "float16":
        ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if device_type == "cuda"
            else torch.npu.amp.autocast(dtype=torch.float16)
        )
    else:  # bfloat16
        ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if device_type == "cuda"
            else torch.npu.amp.autocast(dtype=torch.bfloat16)
        )

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "npu:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_type": args.model_type,
                "language": args.language,
                "vocab_size": vocab_size,
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "hidden_size": args.hidden_size,
                "learning_rate": args.learning_rate,
                "sft_lr": args.sft_lr,
                "dtype": args.dtype,
            },
        )
    else:
        wandb = None

    # Load dataset according to language / 根据语言加载数据集
    if args.language == "zh":
        from dataset.lm_zh_dataset import SFTDataset
    else:
        from dataset.lm_en_dataset import SFTDataset

    # Initialize model (pass model type and language) / 初始化模型（传入模型类型和语言）
    model, tokenizer = init_model(lm_config, args.model_type, args.language)

    tokenizer.pad_token = tokenizer.eos_token
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
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

    scaler = torch.npu.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    # Use sft_lr if provided, otherwise fallback to learning_rate / 如果提供了sft_lr则使用，否则回退到learning_rate
    actual_lr = args.sft_lr if hasattr(args, "sft_lr") else args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr)

    if args.resume:
        weight_filename = f"sft_{args.model_type}_{args.language}_{lm_config.dim}.pth"
        ckp = os.path.join(args.save_dir, weight_filename)
        if os.path.exists(ckp):
            checkpoint = torch.load(ckp, map_location=args.device)
            if isinstance(model, DistributedDataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            Logger(f"Resumed training from: {ckp}!")
        else:
            Logger(f"Warning: Checkpoint {ckp} not found, starting from scratch.")

    if ddp:
        model._ddp_params_and_buffers_to_ignore = (
            {"pos_cis"} if hasattr(model, "pos_cis") else set()
        )
        model = DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    iter_per_epoch = len(train_loader)

    Logger("=" * 80)
    Logger(f"Starting full SFT fine-tuning on NPU...")
    Logger(
        f"Model type: {args.model_type.upper()} | Language: {args.language.upper()} | Vocabulary size: {vocab_size}"
    )
    Logger(f"Log file: {log_path}")
    Logger(f"Configuration:")
    Logger(
        f"  - Pretrained weights: pretrain_{args.model_type}_{args.language}_{args.hidden_size}_final.pth"
    )
    Logger(
        f"  - Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
    )
    Logger(f"  - Batch size: {args.batch_size}")
    Logger(f"  - Learning rate: {actual_lr} (SFT specific)")
    Logger(f"  - Data precision: {args.dtype}")
    Logger("=" * 80)

    for epoch in range(args.epochs):
        Logger(f"\n{'='*60}")
        Logger(f"Starting epoch {epoch+1}/{args.epochs}")
        Logger(f"{'='*60}")
        train_epoch(epoch, wandb)
