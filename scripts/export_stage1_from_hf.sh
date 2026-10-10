#!/usr/bin/env bash
# 将 Hugging Face 格式的指令模型目录转为 parameters/sft_QF_en_1024.pth。
# 前置：该目录下须已有 pytorch_model.bin（自己用 convert_model.py 从 SFT .pth 转出，
# 或自行从 ModelScope 下载官方权重后放入 parameters/QF_Struct_300M/）。
# 若已用 train_full_sft.py 直接得到 sft_QF_en_1024.pth，可跳过本脚本。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF="${HF:-$ROOT/parameters/QF_Struct_300M}"
OUT="${OUT:-$ROOT/parameters/sft_QF_en_1024.pth}"

if [[ ! -f "$HF/pytorch_model.bin" ]]; then
  echo "错误: 未找到 $HF/pytorch_model.bin" >&2
  echo "请先完成通用 SFT 并用 tools/convert_model.py 转成 HF，或自行下载官方权重。" >&2
  exit 1
fi

python3 "$ROOT/tools/convert_model.py" --mode transformers2torch \
  --transformers_path "$HF" \
  --torch_path "$OUT" \
  --model_type QF --language en \
  --hidden_size 1024 --num_hidden_layers 24 \
  --num_attention_heads 16 --num_key_value_heads 4 --max_seq_len 1024

echo "已写出 $OUT"
