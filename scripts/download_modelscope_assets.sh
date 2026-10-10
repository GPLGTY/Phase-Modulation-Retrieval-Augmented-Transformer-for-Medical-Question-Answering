#!/usr/bin/env bash
# 从 ModelScope 按需下载大文件包，并解压到本仓库对应路径。
#
# 用法（在项目根目录）:
#   bash scripts/download_modelscope_assets.sh
#   PACKS="02 04" bash scripts/download_modelscope_assets.sh   # 仅 Intent 评测
#   PACKS="01 04" bash scripts/download_modelscope_assets.sh   # Intent SFT（阶段一权重+数据）
#   PACKS="01 04 05" bash scripts/download_modelscope_assets.sh
#
# 环境变量:
#   MODELSCOPE_REPO   默认 GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
#   DOWNLOAD_DIR      默认 $ROOT/.modelscope_packs
#   PACKS             空格分隔包号：01 02 03a 03b 03c 03d 04 05
#   SKIP_EXTRACT=1    只下载不解压
#   FORCE=1           强制重新下载
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELSCOPE_REPO="${MODELSCOPE_REPO:-GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT/.modelscope_packs}"
PACKS="${PACKS:-01 02 03a 03b 03c 03d 04 05}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FORCE="${FORCE:-0}"

mkdir -p "$DOWNLOAD_DIR"

declare -A PACK_FILE=(
  [01]="01_parameters.tar"
  [02]="02_intent_sft_checkpoints.tar"
  [03a]="03a_corpus_SlimPajama-1B_full.tar"
  [03b]="03b_corpus_openhermes2_5.tar"
  [03c]="03c_corpus_pretrain_hq_zh.tar"
  [03d]="03d_corpus_sft_mini_512_zh.tar"
  [04]="04_medqa_intent_json.tar"
  [05]="05_embedding_stella.tar"
)

if ! command -v modelscope >/dev/null 2>&1; then
  echo "未找到 modelscope CLI，正在安装：pip install -U modelscope" >&2
  pip install -U "modelscope>=1.11.0"
fi

files=()
for key in $PACKS; do
  fname="${PACK_FILE[$key]:-}"
  if [[ -z "$fname" ]]; then
    echo "未知包序号: $key（可选: ${!PACK_FILE[*]}）" >&2
    exit 1
  fi
  files+=("$fname")
done

echo "仓库: $MODELSCOPE_REPO"
echo "缓存: $DOWNLOAD_DIR"
echo "下载: ${files[*]}"

dl_args=(download --repo-type model "$MODELSCOPE_REPO" --local-dir "$DOWNLOAD_DIR")
if [[ "$FORCE" == "1" ]]; then
  dl_args+=(--force)
fi
# 只拉需要的包（外加校验文件，若存在）
modelscope "${dl_args[@]}" "${files[@]}" SHA256SUMS.txt MANIFEST.md 2>/dev/null \
  || modelscope "${dl_args[@]}" "${files[@]}"

if [[ "$SKIP_EXTRACT" == "1" ]]; then
  echo "已跳过解压（SKIP_EXTRACT=1）"
  exit 0
fi

for fname in "${files[@]}"; do
  src="$(find "$DOWNLOAD_DIR" -name "$fname" -type f | head -n 1 || true)"
  if [[ -z "$src" ]]; then
    echo "未找到 $fname（请确认 ModelScope 仓库已包含该文件）" >&2
    echo "仓库: https://www.modelscope.cn/models/$MODELSCOPE_REPO" >&2
    exit 1
  fi
  echo "解压 $src → $ROOT"
  tar -xf "$src" -C "$ROOT"
done

echo
echo "完成。自检示例："
echo "  ls -lh out/run_sft_intent/sft_intent_QF_en_1024_best.pth"
echo "  ls -lh parameters/sft_QF_en_1024.pth"
echo "  ls -lh dataset/RAG_MedQA_US_test_1273_intent.json"
echo "评测："
echo "  CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth bash scripts/eval_medqa_test.sh"
