#!/usr/bin/env bash
# MedQA Intent 二阶段 SFT
# 初始化: parameters/sft_QF_en_1024.pth
# 训练: dataset/RAG_MedQA_US_train_10178_intent.json
# 验证: dataset/RAG_MedQA_US_dev_1272_intent.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INIT="${INIT:-$ROOT/parameters/sft_QF_en_1024.pth}"
OUT="${OUT:-$ROOT/out/run_sft_intent}"
TRAIN="${TRAIN:-$ROOT/dataset/RAG_MedQA_US_train_10178_intent.json}"
VAL="${VAL:-$ROOT/dataset/RAG_MedQA_US_dev_1272_intent.json}"

if [[ ! -f "$INIT" ]]; then
  echo "缺少初始化权重: $INIT" >&2
  echo "请先完成预训练与通用 SFT，或使用仓库已附带的 parameters/sft_QF_en_1024.pth。" >&2
  exit 1
fi

mkdir -p "$OUT"

python3 "$ROOT/trainer/train_sft_intent.py" \
  --init_ckpt "$INIT" \
  --data_path "$TRAIN" \
  --val_data_path "$VAL" \
  --out_dir "$OUT" \
  --model_type QF \
  --language en \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --num_key_value_heads 4 \
  --max_seq_len 4096 \
  --batch_size 6 \
  --accumulation_steps 8 \
  --epochs 6 \
  --sft_lr 5e-5 \
  --early_stop_loss 0 \
  --best_metric val_answer_acc

echo "训练结束: $OUT/sft_intent_QF_en_1024_best.pth"
echo "测试: CHECKPOINT=$OUT/sft_intent_QF_en_1024_best.pth bash scripts/eval_medqa_test.sh"
