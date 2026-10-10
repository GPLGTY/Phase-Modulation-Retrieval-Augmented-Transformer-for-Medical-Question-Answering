#!/usr/bin/env bash
# 在 MedQA Intent 测试集上评测，结果写入 results/ 下自建目录，并与参考 summary 对比。
# 默认评测设置: temperature=0.7, ntk_alpha=0.4
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-$ROOT/out/run_sft_intent/sft_intent_QF_en_1024_best.pth}"
DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_test_1273_intent.json}"
RESULTS="${RESULTS:-$ROOT/results/medqa_intent_test1273_eval}"
TEMPS="${TEMPS:-0.7}"
NTKS="${NTKS:-0.4}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "错误: 找不到权重 $CHECKPOINT" >&2
  echo "请先完成 Intent SFT，或把 CHECKPOINT 指到你的 best.pth" >&2
  exit 1
fi

mkdir -p "$RESULTS"

python3 "$ROOT/scripts/run_ablation_temp_ntk.py" \
  --checkpoint "$CHECKPOINT" \
  --data_path "$DATA" \
  --results_dir "$RESULTS" \
  --temperatures "$TEMPS" \
  --ntk_alphas "$NTKS" \
  --max_seq_len 4096 \
  --ntk_max_train_len 4096 \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --num_key_value_heads 4 \
  --dtype bfloat16 \
  --max_new_tokens 1 \
  --batch_size 1

echo
echo "本次结果: $RESULTS/summary.csv"
echo "参考结果: $ROOT/results/medqa_intent_test1273_reference/summary.csv"
