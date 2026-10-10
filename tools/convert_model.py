"""
Model conversion script: convert between torch format (state_dict) and HuggingFace Transformers format.
模型转换脚本：在 torch 格式（state_dict）与 HuggingFace Transformers 格式之间转换。

Supports two model types: QF (QubitFormer) and MM (MiniMindForCausalLM).
支持两种模型类型：QF（QubitFormer）和 MM（MiniMindForCausalLM）。

Supports two languages: zh (Chinese) and en (English), which affect tokenizer and vocabulary size.
支持两种语言：zh（中文）和 en（英文），影响分词器与词汇表大小。
"""

import torch
import warnings
import sys
import os
import argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from model.QFConfig import QFConfig
from model.model import MiniMindForCausalLM
from model.qbmodel import QubitFormer

warnings.filterwarnings("ignore", category=UserWarning)


def export_tokenizer(transformers_path, language):
    """
    Save tokenizer to the target directory based on language.
    根据语言将分词器保存到目标目录。

    Args:
        transformers_path (str): Directory to save tokenizer / 保存分词器的目录
        language (str): 'zh' or 'en' / 语言选择
    """
    if language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("../model/tokenizer_zh")
    else:  # en
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "../model/tokenizer_en",
            cache_dir="../model/tokenizer_en",
            local_files_only=True,
        )
    tokenizer.save_pretrained(transformers_path)
    print(f"Tokenizer saved to: {transformers_path}")


def convert_torch2transformers(
    torch_path, transformers_path, lm_config, model_type, language
):
    """
    Convert PyTorch state_dict (.pth) to HuggingFace Transformers model format.
    将 PyTorch state_dict（.pth）转换为 HuggingFace Transformers 模型格式。

    Args:
        torch_path (str): Path to input .pth file / 输入的 .pth 文件路径
        transformers_path (str): Directory to save Transformers model / 保存 Transformers 模型的目录
        lm_config (QFConfig): Model configuration / 模型配置
        model_type (str): 'QF' or 'MM' / 模型类型
        language (str): 'zh' or 'en' / 语言
    """
    # Instantiate model based on model_type / 根据模型类型实例化模型
    if model_type == "QF":
        # Register QubitFormer for auto classes / 注册 QubitFormer 到自动类
        QFConfig.register_for_auto_class()
        QubitFormer.register_for_auto_class("AutoModelForCausalLM")
        model = QubitFormer(lm_config)
    else:  # MM
        # Register MiniMindForCausalLM for auto classes / 注册 MiniMindForCausalLM 到自动类
        QFConfig.register_for_auto_class()  # Using QFConfig for MM as well / MM 也使用 QFConfig
        MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        model = MiniMindForCausalLM(lm_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(torch_path, map_location=device)

    # Load weights, allow missing/unexpected keys for compatibility / 加载权重，允许缺失/多余键以兼容
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys (will be randomly initialized): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters: {model_params / 1e6:.3f} million = {model_params / 1e9:.3f} B (Billion)"
    )

    # Save model and tokenizer / 保存模型和分词器
    model.save_pretrained(transformers_path, safe_serialization=False)
    export_tokenizer(transformers_path, language)
    print(f"Model saved as Transformers format: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path, model_type, language):
    """
    Convert HuggingFace Transformers model to PyTorch state_dict (.pth).
    将 HuggingFace Transformers 模型转换为 PyTorch state_dict（.pth）。

    Args:
        transformers_path (str): Directory of Transformers model / Transformers 模型目录
        torch_path (str): Output .pth file path / 输出的 .pth 文件路径
        model_type (str): 'QF' or 'MM' (used for trust_remote_code) / 模型类型（用于 trust_remote_code）
        language (str): Not directly used, but kept for consistency / 不直接使用，保留以保持接口一致
    """
    model = AutoModelForCausalLM.from_pretrained(
        transformers_path, trust_remote_code=True
    )
    torch.save(model.state_dict(), torch_path)
    print(f"Model saved as PyTorch format: {torch_path}")


def push_to_hf(export_model_path, model_type, language):
    """
    (Optional) Push model to HuggingFace Hub.
    （可选）将模型推送到 HuggingFace Hub。
    """
    tokenizer = None
    if language == "zh":
        tokenizer = AutoTokenizer.from_pretrained("../model/tokenizer_zh")
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "../model/tokenizer_en",
            cache_dir="../model/tokenizer_en",
            local_files_only=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        export_model_path, trust_remote_code=True
    )
    # Uncomment below to push to hub / 取消注释以下行以推送到 hub
    # model.push_to_hub(export_model_path)
    # tokenizer.push_to_hub(export_model_path, safe_serialization=False)
    print("Push to HF Hub is commented out. Uncomment if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between torch and Transformers formats for MiniMind models."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="torch2transformers",
        choices=["torch2transformers", "transformers2torch"],
        help="Conversion mode: torch -> Transformers or Transformers -> torch",
    )
    parser.add_argument(
        "--torch_path",
        type=str,
        required=True,
        help="Path to input/output .pth file (depending on mode)",
    )
    parser.add_argument(
        "--transformers_path",
        type=str,
        required=True,
        help="Path to input/output Transformers directory (depending on mode)",
    )

    # Model type and language / 模型类型与语言
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

    # Model configuration parameters / 模型配置参数
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden dimension / 隐藏层维度"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=24,
        help="Number of hidden layers / 隐藏层数量",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads / 注意力头数",
    )
    parser.add_argument(
        "--num_key_value_heads",
        type=int,
        default=2,
        help="Number of key-value heads for GQA / GQA 的 KV 头数",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length / 最大序列长度",
    )

    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=True,
        help="Use Flash Attention / 使用 Flash Attention",
    )

    args = parser.parse_args()

    # Set vocabulary size based on language / 根据语言设置词汇表大小
    if args.language == "zh":
        vocab_size = 6400
    else:  # en
        vocab_size = 32000

    # Build model configuration / 构建模型配置
    lm_config = QFConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
        flash_attn=args.flash_attn,
    )

    print("=" * 60)
    print(f"Conversion mode: {args.mode}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Language: {args.language.upper()} (vocab_size={vocab_size})")
    print(
        f"Model config: hidden_size={args.hidden_size}, n_layers={args.num_hidden_layers}, "
        f"n_heads={args.num_attention_heads}, n_kv_heads={args.num_key_value_heads}, "
        f"max_seq_len={args.max_seq_len}"
    )
    print("=" * 60)

    if args.mode == "torch2transformers":
        convert_torch2transformers(
            torch_path=args.torch_path,
            transformers_path=args.transformers_path,
            lm_config=lm_config,
            model_type=args.model_type,
            language=args.language,
        )
    else:  # transformers2torch
        convert_transformers2torch(
            transformers_path=args.transformers_path,
            torch_path=args.torch_path,
            model_type=args.model_type,
            language=args.language,
        )
