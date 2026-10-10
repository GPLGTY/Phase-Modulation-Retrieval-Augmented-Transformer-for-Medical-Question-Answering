# Phase-modulation-transforme

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

中文说明：[README.md](README.md) / [README_CN.md](README_CN.md)

GitHub: https://github.com/GPLGTY/Phase-modulation-transforme

This repository provides the full training and evaluation pipeline for a phase-modulation retrieval-augmented Transformer on MedQA: code, data scripts, and reference results. **Large weights / corpora / embeddings are hosted on ModelScope** (see below).

> Training uses seed `1337`; tiny GPU nondeterminism may remain. After the same recipe, test accuracy should be near the reference.

## Environment

```bash
cd Phase-modulation-transforme
pip install -r requirements.txt
```

Install a CUDA-matched PyTorch. For Ascend, install `torch_npu` (see comments in `requirements.txt`).

Optional:

```bash
pip install lm-eval==0.4.9
```

### Core packages

See [requirements.txt](./requirements.txt). Main dependencies: Python 3.8+, torch ≥2.0, transformers ≥4.37, datasets, wandb, tqdm. Optional: `modelscope` for asset download.

## Download weights & data from ModelScope

GitHub holds code only. **All weights, corpora, and embeddings are on ModelScope** (all `01`–`05` packs are published).

**ModelScope:** [`GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`](https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
Details: [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md).

### One-shot download + extract

From the **project root**:

```bash
pip install -U modelscope

# Full (~17G)
bash scripts/download_modelscope_assets.sh

# Intent eval only (~1.7G): checkpoints + MedQA Intent JSON
PACKS="02 04" bash scripts/download_modelscope_assets.sh

# Intent SFT: stage-1 weights + Intent data
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# Plus Stella embeddings for the retrieval pipeline
PACKS="01 04 05" bash scripts/download_modelscope_assets.sh
```

Or download `.tar` from the ModelScope page and run `tar -xf <pack>.tar` in the project root.

### Pack → local path

| ModelScope file | Extracts to | Use |
|-----------------|-------------|-----|
| `01_parameters.tar` | `parameters/QF_*`, `MiniMind_*`, `sft_QF_en_1024.pth` | HF weights + stage-1 English SFT |
| `02_intent_sft_checkpoints.tar` | `out/run_sft_intent/*_best.pth` (and last) | Intent SFT; **eval with best** |
| `03a`–`03d` corpus tars | `dataset/*.jsonl` | Pretrain / general SFT corpora |
| `04_medqa_intent_json.tar` | `dataset/RAG_MedQA_US_*_intent.json` | Intent train/val/test |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | Stella embeddings for retrieval |

### After download

```bash
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh

# optional Intent SFT (after PACKS="01 04")
# bash scripts/run_stage2_intent_sft.sh
```

## Layout

```
Phase-modulation-transforme/
├── model/
├── trainer/            # pretrain / general SFT / MedQA Intent SFT
├── tools/
├── scripts/
├── dataset/
│   ├── pipeline/       # MedQA data-generation scripts + embedding weights
│   ├── pipeline_data/  # intermediate data (created when you run the pipeline)
│   └── *.json(l)       # pretrain/SFT corpora + MedQA Intent files
├── parameters/
├── out/run_sft_intent/
├── results/
│   └── medqa_intent_test1273_reference/
├── requirements.txt
└── LICENSE
```

## Data

### Data files (after ModelScope extract; skip generation if present)

| Stage | Path | Pack |
|-------|------|------|
| English pretrain | `dataset/SlimPajama-1B_full.jsonl` | `03a` |
| English general SFT | `dataset/openhermes2_5.jsonl` | `03b` |
| Chinese pretrain (optional) | `dataset/pretrain_hq_zh.jsonl` | `03c` |
| Chinese SFT (optional) | `dataset/sft_mini_512_zh.jsonl` | `03d` |
| MedQA Intent train | `dataset/RAG_MedQA_US_train_10178_intent.json` (10178) | `04` |
| MedQA Intent val | `dataset/RAG_MedQA_US_dev_1272_intent.json` (1272) | `04` |
| MedQA Intent test | `dataset/RAG_MedQA_US_test_1273_intent.json` (1273) | `04` |

Public corpus backup: https://www.modelscope.cn/datasets/longview/QubitFormerDataset  

If MedQA Intent files are **missing**, or you need to **rebuild from raw MedQA**, follow **MedQA dataset** and **Dataset preprocessing** below. Details: [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md), [dataset/pipeline/README.md](dataset/pipeline/README.md).

### MedQA dataset (download)

- Source: https://github.com/jind11/MedQA  
- Download: https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing  
- Intro (zh): https://zhuanlan.zhihu.com/p/679590312  

After unpacking, place textbooks and questions under:

```
dataset/pipeline_data/
├── textbooks/     # en / zh_paragraph / zh_sentence (txt)
└── questions/     # US|Mainland with train.jsonl / dev.jsonl / test.jsonl
```

### MedMCQA dataset (download)

- Homepage: https://medmcqa.github.io/  
- Source: https://github.com/medmcqa/medmcqa  
- Download: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky  
- Hugging Face: https://huggingface.co/datasets/openlifescienceai/medmcqa  
- Paper: https://arxiv.org/abs/2203.14371  

After download, place `train.json` / `dev.json` under `dataset/pipeline_data/questions/MedMCQA/`. In step 3, convert with `convert_medmcqa_to_medqa_jsonl.py` to MedQA-style jsonl, then retrieve.

Embedding checkpoints go under `dataset/pipeline/models/` via ModelScope pack `05` (`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`). `dataset/pipeline/config.py` points there; update `model_path` if you move them.

For optional LLM explanation generation, set API keys, e.g.:

```bash
export KIMI_OPENAI_KEY='xxxx'
```

### Dataset preprocessing (generate from scratch)

Run all commands under `dataset/pipeline/`. Single GPU: `--device_ids="0"` and `--num_process=1`.  
Steps 3–4 use `--query_key_name="question_with_options"` only. Do not run `question` then `question_with_options` into the same output path (overwrite risk).

#### 1. Convert textbook txt → json

```bash
cd dataset/pipeline

python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

#### 2. Build vector stores

```bash
cd dataset/pipeline

python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'
```

#### 3. Retrieve knowledge for train / dev / test

Always use `--query_key_name="question_with_options"` (aligned with step 4). Output path:

`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

Single GPU: `--device_ids="0"` and `--num_process=1`. Input must be **jsonl** (one JSON per line).

```bash
cd dataset/pipeline

# ---------- MedQA English US (main Intent path) ----------
python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/test.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/test" \
    --output_json_file="test.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

# ---------- MedQA Chinese Mainland (optional) ----------
python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

# ---------- MedMCQA: convert format, then retrieve (train / val) ----------
# Raw MedMCQA uses opa/opb/opc/opd + cop; convert to MedQA-style jsonl first
# Place downloaded train.json / dev.json under ../pipeline_data/questions/MedMCQA/
python convert_medmcqa_to_medqa_jsonl.py \
  --in_jsonl ../pipeline_data/questions/MedMCQA/train.json \
  --out_jsonl ../pipeline_data/questions/MedMCQA/train.jsonl

python convert_medmcqa_to_medqa_jsonl.py \
  --in_jsonl ../pipeline_data/questions/MedMCQA/dev.json \
  --out_jsonl ../pipeline_data/questions/MedMCQA/dev.jsonl

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/MedMCQA/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/MedMCQA/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/MedMCQA/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/MedMCQA/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"
```

#### 4. Retrieval JSON → Intent training data

Complete step 3 first. Run `dataset/pipeline/generate_intent_from_retrieval.py` to rewrite retrieval JSON (`question_with_options`, `retrieve_info`, …) into Intent samples (Intent style) via an **OpenAI-compatible API**. Supports resume via `*.partial.json`.

**API key**: never commit secrets. Use either:

```bash
export VOLCENGINE_ARK_API_KEY='your_key'   # or ARK_API_KEY
# or pass --api_key 'your_key'
```

Default `base_url` is Volcengine Ark. You can also serve Distill models locally with **vLLM** (OpenAI-compatible `/v1`); see below.

`--in_json` must be step-3 `question_with_options` output. Optionally add `--with_rag` to put retrieved passages in the LLM prompt.

##### Local DeepSeek Distill (vLLM) for Intent generation

Works for `DeepSeek-R1-Distill-Qwen-32B`, `DeepSeek-R1-Distill-Llama-70B`, and similar. Download weights from Hugging Face or ModelScope, e.g.:

| Model | Hugging Face | ModelScope (example) |
|-------|----------------|----------------------|
| Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| Distill-Llama-70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |

**GPU memory (rough, bf16, unquantized):** Qwen-32B ≈ **1–2×80GB**; Llama-70B ≈ **2–4×80GB** (use `--tensor-parallel-size`). For smaller GPUs, use AWQ/GPTQ (or similar) and adjust launch flags.

**1) Start vLLM (OpenAI-compatible server)**

```bash
pip install -U vllm

python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192

# 70B example (more GPUs)
# python -m vllm.entrypoints.openai.api_server \
#   --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
#   --host 0.0.0.0 --port 8000 \
#   --dtype bfloat16 \
#   --tensor-parallel-size 4 \
#   --max-model-len 8192
```

Then set `--base_url http://127.0.0.1:8000/v1`, any non-empty `--api_key` (e.g. `EMPTY`), and `--model` matching the vLLM `--model` (or served-model-name).

**2) Generate Intent with the local server (MedQA US)**

```bash
cd dataset/pipeline

python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedQA_US_train_intent.json \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --workers 4 --temperature 0.7 --max_tokens 4096

# Val / test: only change --in_json / --out_json (same paths as the cloud examples below)
```

**3) MedMCQA example (local vLLM)**

```bash
cd dataset/pipeline

python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedMCQA_train_intent.json \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --workers 2 --temperature 0.7 --max_tokens 4096
```

Keep `--workers` moderate so the GPU queue is not saturated.

**MedQA (English US) — train / val / test** (after step-3 US retrieval; cloud API examples — for local vLLM, set `--base_url` / `--api_key` / `--model` as above)

```bash
cd dataset/pipeline

# Train
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedQA_US_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# Validation
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
  --out_json ../RAG_MedQA_US_dev_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# Test
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test.json \
  --out_json ../RAG_MedQA_US_test_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096
```

**MedMCQA — train / validation** (after step-3 MedMCQA convert + retrieve; cloud API examples)

```bash
cd dataset/pipeline

# Train
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedMCQA_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# Validation
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
  --out_json ../RAG_MedMCQA_dev_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096
```

| Flag | Meaning |
|------|---------|
| `--in_json` | Retrieval sample-list JSON (required) |
| `--out_json` | Output Intent JSON (required) |
| `--with_rag` | Put retrieved passages into the prompt |
| `--offset` / `--n` | Slice; `--n 0` = all after offset |
| `--model` / `--base_url` / `--api_key` | Model and auth |
| `--workers` | Parallelism |
| `--no_resume` | Ignore existing partial and restart |

If you already extracted ModelScope pack `04` (`RAG_MedQA_US_*_intent.json`), skip this step and go to Intent SFT. See also [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md) and [dataset/pipeline/README.md](dataset/pipeline/README.md).

### Weights (after ModelScope extract)

| Path | Description | Pack |
|------|-------------|------|
| `parameters/QF_Base_300M/pytorch_model.bin` | QF Base | `01` |
| `parameters/QF_Struct_300M/pytorch_model.bin` | QF Struct | `01` |
| `parameters/MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base | `01` |
| `parameters/MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct | `01` |
| `parameters/sft_QF_en_1024.pth` | English general SFT (stage 1) | `01` |
| `out/run_sft_intent/sft_intent_QF_en_1024_best.pth` | Intent SFT best (recommended for eval) | `02` |
| `out/run_sft_intent/sft_intent_QF_en_1024.pth` | Intent SFT last epoch | `02` |

You may also train these checkpoints from scratch as below.

## Pretraining (optional, from scratch)

```bash
cd trainer
python train_pretrain.py \
    --model_type QF \
    --language en \
    --hidden_size 1024 \
    --num_hidden_layers 24 \
    --num_attention_heads 8 \
    --num_key_value_heads 2 \
    --max_seq_len 1024 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --epochs 1 \
    --data_path ../dataset/SlimPajama-1B_full.jsonl \
    --out_dir ../parameters
```

On Ascend use `train_pretrain_npu.py`. Output: `parameters/pretrain_QF_en_1024_final.pth`.

Optional HF conversion:

```bash
cd ../tools
python convert_model.py --mode torch2transformers \
    --torch_path ../parameters/pretrain_QF_en_1024_final.pth \
    --transformers_path ../parameters/QF_Base_300M_mine \
    --model_type QF --language en \
    --hidden_size 1024 --num_hidden_layers 24 \
    --num_attention_heads 8 --num_key_value_heads 2 --max_seq_len 1024
```

## General SFT (stage 1, optional)

If you already downloaded pack `01` (`parameters/sft_QF_en_1024.pth`), skip this section and go to Intent SFT.

```bash
cd trainer
python train_full_sft.py \
    --model_type QF \
    --language en \
    --hidden_size 1024 \
    --num_hidden_layers 24 \
    --num_attention_heads 8 \
    --num_key_value_heads 2 \
    --max_seq_len 1024 \
    --batch_size 100 \
    --learning_rate 5e-4 \
    --sft_lr 1e-4 \
    --epochs 2 \
    --data_path ../dataset/openhermes2_5.jsonl \
    --out_dir ../parameters
```

Produces `parameters/sft_QF_en_1024.pth`. For Chinese, use `--language zh` and `../dataset/sft_mini_512_zh.jsonl`.

Optional: `bash scripts/export_stage1_from_hf.sh`.

## MedQA Intent SFT (stage 2)

Continue from `parameters/sft_QF_en_1024.pth`:

```bash
bash scripts/run_stage2_intent_sft.sh
```

Equivalent:

```bash
python trainer/train_sft_intent.py \
  --init_ckpt parameters/sft_QF_en_1024.pth \
  --data_path dataset/RAG_MedQA_US_train_10178_intent.json \
  --val_data_path dataset/RAG_MedQA_US_dev_1272_intent.json \
  --out_dir out/run_sft_intent \
  --model_type QF --language en \
  --hidden_size 1024 --num_hidden_layers 24 \
  --num_attention_heads 16 --num_key_value_heads 4 \
  --max_seq_len 4096 \
  --batch_size 6 --accumulation_steps 8 \
  --epochs 6 --sft_lr 5e-5 \
  --early_stop_loss 0 --best_metric val_answer_acc
```

Best checkpoint: `out/run_sft_intent/sft_intent_QF_en_1024_best.pth` (from ModelScope pack `02`).

## Test evaluation

Test file: `dataset/RAG_MedQA_US_test_1273_intent.json` (1273 items; `messages` are system/user only; gold label is top-level `answer_idx`; from ModelScope pack `04`).

### How to run

```bash
PACKS="02 04" bash scripts/download_modelscope_assets.sh

CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh
```

Explicit paths:

```bash
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
DATA=dataset/RAG_MedQA_US_test_1273_intent.json \
RESULTS=results/medqa_intent_test1273_eval \
TEMPS=0.7 NTKS=0.4 \
  bash scripts/eval_medqa_test.sh
```

Defaults: `temperature=0.7`, `ntk_alpha=0.4`.

Full grid:

```bash
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
TEMPS=0.7,0.9,1.1,1.3,1.5 NTKS=0.4,0.6,0.8,1.0,1.2 \
  bash scripts/eval_medqa_test.sh
```

### How to read results

1. Your run: `results/medqa_intent_test1273_eval/summary.csv`
2. Reference: `results/medqa_intent_test1273_reference/summary.csv`

| Column | Meaning |
|--------|---------|
| `temperature` / `ntk_alpha` | Inference hyperparameters |
| `n_total` / `n_scored` | Sample count (should be 1273) |
| `n_match` | Predictions matching `answer_idx` |
| `accuracy` | `n_match / n_scored` |
| `output_csv` | Per-example CSV |

```bash
python -c "import pandas as pd; print(pd.read_csv('results/medqa_intent_test1273_eval/summary.csv')[['temperature','ntk_alpha','n_match','accuracy']])"
```

### Optional paper-style eval

```bash
lm-eval --model hf \
  --model_args pretrained=./parameters/QF_Struct_300M,trust_remote_code=True \
  --tasks hellaswag,boolq,piqa,arc_easy,winogrande,openbookqa,arc_challenge \
  --device cuda:0 --batch_size 8
```

## Recommended path (quick eval)

```bash
pip install -r requirements.txt
pip install -U modelscope

PACKS="02 04" bash scripts/download_modelscope_assets.sh

CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh
```
