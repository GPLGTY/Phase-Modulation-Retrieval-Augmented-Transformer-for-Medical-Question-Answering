# Phase-Modulation Retrieval-Augmented Transformer for Medical Question Answering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

English: [README_EN.md](README_EN.md)

本仓库提供相位调制检索增强 Transformer 在 MedQA 医学问答上的完整训练与评测流程：代码、数据处理脚本、测试参考结果；**大体积权重 / 语料 / Embedding 请从 ModelScope 下载**（见下文）。参考最优 **1197/1273 ≈ 0.9403**（`temperature=0.7`，`ntk_alpha=0.4`）；复现实测 best 约 **1199/1273 ≈ 0.9419**。

> 训练使用固定随机种子 `1337`，仍可能因 GPU 非确定性出现极小数值差；完整配方跑通后，测试准确率应落在参考结果附近。

## 项目环境依赖

```bash
cd Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
pip install -r requirements.txt
```

请安装与本机 CUDA 匹配的 PyTorch。昇腾环境再安装 `torch_npu`（见 `requirements.txt` 注释）。

可选下游评测：

```bash
pip install lm-eval==0.4.9
```

### 软件包版本依赖

完整依赖见 [requirements.txt](./requirements.txt)。核心：

| 软件包 | 说明 |
|--------|------|
| python | 3.8+ |
| torch | ≥2.0，需匹配本机 CUDA |
| transformers | ≥4.37 |
| datasets / wandb / tqdm | 训练与日志 |
| modelscope | 可选，用于下载大文件包 |

## 从 ModelScope 下载权重与数据

本仓库（GitHub）主要放代码；**权重、语料、Embedding 已全部放在 ModelScope**，克隆代码后按需下载即可。

**ModelScope**：[`GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`](https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
（已含 `01`–`05` 全部 `.tar` 及 `SHA256SUMS.txt` / `MANIFEST.md`。更细说明见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)。）

### 一键下载并解压

在**项目根目录**执行（自动解压到 `parameters/`、`out/`、`dataset/` 等正确路径）：

```bash
pip install -U modelscope   # 首次需要

# 全量（约 17G）
bash scripts/download_modelscope_assets.sh

# 仅 Intent 测试评测（约 1.7G）：Intent 权重 + MedQA Intent JSON
PACKS="02 04" bash scripts/download_modelscope_assets.sh

# Intent SFT：阶段一英文 SFT 权重 + Intent 数据
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 还需本地重跑检索流水线时加上 Embedding
PACKS="01 04 05" bash scripts/download_modelscope_assets.sh
```

也可在 ModelScope 网页手动下载 `.tar`，于项目根目录执行 `tar -xf <包名>.tar`。

### 包内容与放置位置

| ModelScope 文件 | 解压后位置 | 用途 |
|-----------------|------------|------|
| `01_parameters.tar` | `parameters/QF_*`、`parameters/MiniMind_*`、`parameters/sft_QF_en_1024.pth` | HF 权重 + 阶段一英文 SFT |
| `02_intent_sft_checkpoints.tar` | `out/run_sft_intent/sft_intent_QF_en_1024_best.pth`（及 last） | Intent SFT；**评测用 best** |
| `03a_corpus_SlimPajama-1B_full.tar` | `dataset/SlimPajama-1B_full.jsonl` | 英文预训练语料 |
| `03b_corpus_openhermes2_5.tar` | `dataset/openhermes2_5.jsonl` | 英文通用 SFT 语料 |
| `03c_corpus_pretrain_hq_zh.tar` | `dataset/pretrain_hq_zh.jsonl` | 中文预训练（可选） |
| `03d_corpus_sft_mini_512_zh.tar` | `dataset/sft_mini_512_zh.jsonl` | 中文 SFT（可选） |
| `04_medqa_intent_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*_intent.json` | Intent 训练/验证/测试 |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | 检索用 Stella Embedding |

### 下载后怎么用

```bash
# 1) 评测（需先 PACKS="02 04"）
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh

# 2) Intent SFT（需先 PACKS="01 04"）
bash scripts/run_stage2_intent_sft.sh
```

## 目录结构

```
Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering/
├── model/              # 模型与分词器
├── trainer/            # 预训练 / 通用 SFT / MedQA Intent SFT
├── tools/              # 权重转换、长度外推
├── scripts/            # Intent SFT、测试评测
├── dataset/
│   ├── pipeline/       # MedQA 数据生成脚本 + Embedding 权重
│   ├── pipeline_data/  # 生成时的中间数据（运行后出现，可不入库）
│   └── *.json(l)       # 预训练/SFT 语料与 MedQA Intent 成品
├── parameters/         # HF 权重与通用 SFT 检查点
├── out/run_sft_intent/ # Intent SFT 检查点
├── results/
│   └── medqa_intent_test1273_reference/   # 测试参考结果
├── requirements.txt
└── LICENSE
```

## 数据

### 数据文件（`PACKS` 含对应包并解压后即有）

| 阶段 | 路径 | 包 |
|------|------|-----|
| 英文预训练 | `dataset/SlimPajama-1B_full.jsonl` | `03a` |
| 英文通用 SFT | `dataset/openhermes2_5.jsonl` | `03b` |
| 中文预训练（可选） | `dataset/pretrain_hq_zh.jsonl` | `03c` |
| 中文 SFT（可选） | `dataset/sft_mini_512_zh.jsonl` | `03d` |
| MedQA Intent 训练 | `dataset/RAG_MedQA_US_train_10178_intent.json`（10178） | `04` |
| MedQA Intent 验证 | `dataset/RAG_MedQA_US_dev_1272_intent.json`（1272） | `04` |
| MedQA Intent 测试 | `dataset/RAG_MedQA_US_test_1273_intent.json`（1273） | `04` |

下载：`PACKS="04"` 或与权重一起 `PACKS="02 04"` / `PACKS="01 04"`。公开语料另有备份：https://www.modelscope.cn/datasets/longview/QubitFormerDataset  

若要从原始 MedQA **重新生成** Intent 数据，见下面「MedQA 数据集」与「数据集预处理」；说明见 [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md)、[dataset/pipeline/README.md](dataset/pipeline/README.md)。

### MedQA 数据集（下载）

- 数据集来源：https://github.com/jind11/MedQA  
- 下载链接：https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing  
- 中文介绍：https://zhuanlan.zhihu.com/p/679590312  

下载并解压后，将教材与题目整理到：

```
dataset/pipeline_data/
├── textbooks/     # en / zh_paragraph / zh_sentence（txt）
└── questions/     # US|Mainland 下的 train.jsonl / dev.jsonl / test.jsonl
```

### MedMCQA 数据集（下载）

- 官网：https://medmcqa.github.io/  
- 数据集来源：https://github.com/medmcqa/medmcqa  
- 下载链接：https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky  
- Hugging Face：https://huggingface.co/datasets/openlifescienceai/medmcqa  
- 论文：https://arxiv.org/abs/2203.14371  

下载并解压后，将 `train.json` / `dev.json` 放到 `dataset/pipeline_data/questions/MedMCQA/`；第 3 步先用 `convert_medmcqa_to_medqa_jsonl.py` 转为 MedQA 格式 jsonl，再检索。

检索用 Embedding 已放在 `dataset/pipeline/models/`（`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`）。`dataset/pipeline/config.py` 已指向这些路径；若你改了存放位置，请同步改 `model_path`。

生成解析（可选）时需配置 LLM API 环境变量，例如：

```bash
export KIMI_OPENAI_KEY='xxxx'
# 以及你实际调用的其它服务 KEY
```

### 数据集预处理（从零生成）

以下命令均在 `dataset/pipeline/` 下执行。单卡时将 `--device_ids="0"` 且 `--num_process=1`。  
第 3–4 步统一使用 `--query_key_name="question_with_options"`（题干+选项）；勿先跑 `question` 再跑 `question_with_options` 到同一输出路径，以免覆盖。

#### 1. 将 txt 教材转换为 json

```bash
cd dataset/pipeline

python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

#### 2. 读取 json 完成向量化

```bash
cd dataset/pipeline

python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'
```

#### 3. 对 train / dev / test 做检索，构建带知识的指令素材

统一使用 `--query_key_name="question_with_options"`（与第 4 步对齐）。产物路径为：

`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

单卡时：`--device_ids="0"` 且 `--num_process=1`。输入须为 **jsonl**（每行一个 JSON）。

```bash
cd dataset/pipeline

# ---------- MedQA 英文 US（Intent 主路线）----------
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

# ---------- MedQA 中文 Mainland（可选）----------
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

# ---------- MedMCQA：先格式转换，再检索（训练 / 验证）----------
# 原始 MedMCQA 为 opa/opb/opc/opd + cop，需先转为与 MedQA 相同的 jsonl 字段
# 将下载的 train.json / dev.json 放到 ../pipeline_data/questions/MedMCQA/
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

#### 4. 检索结果 → Intent 训练数据

须先完成第 3 步。用 `dataset/pipeline/generate_intent_from_retrieval.py`，将检索 JSON（含 `question_with_options`、`retrieve_info` 等）经 **OpenAI 兼容 API** 改写为 Intent 样本。支持断点续跑（`*.partial.json`）。

**密钥**：请勿写入仓库。任选其一：

```bash
export VOLCENGINE_ARK_API_KEY='你的密钥'   # 或 ARK_API_KEY
# 也可在命令行传 --api_key '你的密钥'
```

默认 `base_url` 为火山方舟；本地 vLLM / Ollama 等可改 `--base_url`（常见为 `.../v1`），`--api_key` 可填非空占位如 `EMPTY`。

`--in_json` 必须指向第 3 步 `question_with_options` 产物。可选：加 `--with_rag`，把检索知识写入 LLM 提示。

**MedQA（英文 US）— 训练 / 验证 / 测试**（承接第 3 步 US 检索）

```bash
cd dataset/pipeline

# 训练集
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedQA_US_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# 验证集
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
  --out_json ../RAG_MedQA_US_dev_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# 测试集
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test.json \
  --out_json ../RAG_MedQA_US_test_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096
```

**MedMCQA — 训练集 / 验证集**（承接第 3 步 MedMCQA 转换 + 检索）

```bash
cd dataset/pipeline

# 训练集
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedMCQA_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096

# 验证集
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
  --out_json ../RAG_MedMCQA_dev_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8 --temperature 0.7 --max_tokens 4096
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--in_json` | 检索后的样本列表 JSON（必填） |
| `--out_json` | 输出 Intent JSON（必填） |
| `--with_rag` | 将检索到的段落放入 prompt（仅影响生成，不改变检索文件） |
| `--offset` / `--n` | 切片；`--n 0` 表示从 offset 起全部 |
| `--model` / `--base_url` / `--api_key` | 模型与鉴权 |
| `--workers` | 并发数 |
| `--no_resume` | 忽略已有 partial，从头生成 |

若已从 ModelScope 解压 `04_medqa_intent_json.tar`（或全量包），可跳过本步，直接用于下文 Intent SFT。更多说明见 [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md)、[dataset/pipeline/README.md](dataset/pipeline/README.md)。

### 权重（从 ModelScope 下载）

先执行 `PACKS="01"` / `PACKS="02"` 或全量下载脚本，解压后路径如下：

| 路径 | 说明 | 包 |
|------|------|-----|
| `parameters/QF_Base_300M/pytorch_model.bin` | QF Base | `01` |
| `parameters/QF_Struct_300M/pytorch_model.bin` | QF Struct | `01` |
| `parameters/MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base | `01` |
| `parameters/MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct | `01` |
| `parameters/sft_QF_en_1024.pth` | 英文通用 SFT（阶段一） | `01` |
| `out/run_sft_intent/sft_intent_QF_en_1024_best.pth` | Intent SFT 验证最优（推荐评测） | `02` |
| `out/run_sft_intent/sft_intent_QF_en_1024.pth` | Intent SFT 最后一轮 | `02` |

也可按下文从零训练生成上述检查点。

## 预训练（可选，从头）

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

昇腾用 `train_pretrain_npu.py`。完成后得到 `parameters/pretrain_QF_en_1024_final.pth`。

转成 Hugging Face 目录（可选）：

```bash
cd ../tools
python convert_model.py --mode torch2transformers \
    --torch_path ../parameters/pretrain_QF_en_1024_final.pth \
    --transformers_path ../parameters/QF_Base_300M_mine \
    --model_type QF --language en \
    --hidden_size 1024 --num_hidden_layers 24 \
    --num_attention_heads 8 --num_key_value_heads 2 --max_seq_len 1024
```

## 通用 SFT（阶段一，可选）

若已使用仓库附带的 `parameters/sft_QF_en_1024.pth`，可跳过本节，直接进入 Intent SFT。

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

得到 `parameters/sft_QF_en_1024.pth`。中文则改 `--language zh`，数据用 `../dataset/sft_mini_512_zh.jsonl`。

也可从 HF 目录导出：`bash scripts/export_stage1_from_hf.sh`。

## MedQA Intent SFT（阶段二）

在 `parameters/sft_QF_en_1024.pth` 上继续全参微调：

```bash
bash scripts/run_stage2_intent_sft.sh
```

等价命令：

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

Best 权重：`out/run_sft_intent/sft_intent_QF_en_1024_best.pth`（从 ModelScope 包 `02` 解压后评测）。

## 测试集评测

测试数据：`dataset/RAG_MedQA_US_test_1273_intent.json`（1273 条；`messages` 仅 system/user，标准答案见顶层 `answer_idx`；来自 ModelScope 包 `04`）。

### 怎么跑

先下载权重与测试数据，再评测：

```bash
PACKS="02 04" bash scripts/download_modelscope_assets.sh

CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh
```

显式指定路径：

```bash
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
DATA=dataset/RAG_MedQA_US_test_1273_intent.json \
RESULTS=results/medqa_intent_test1273_eval \
TEMPS=0.7 NTKS=0.4 \
  bash scripts/eval_medqa_test.sh
```

默认：`temperature=0.7`，`ntk_alpha=0.4`（与参考最优一致）。

完整参数网格：

```bash
CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
TEMPS=0.7,0.9,1.1,1.3,1.5 NTKS=0.4,0.6,0.8,1.0,1.2 \
  bash scripts/eval_medqa_test.sh
```

### 结果怎么看

1. 本次汇总：`results/medqa_intent_test1273_eval/summary.csv`
2. 参考汇总：`results/medqa_intent_test1273_reference/summary.csv`

| 列 | 含义 |
|----|------|
| `temperature` / `ntk_alpha` | 推理超参 |
| `n_total` / `n_scored` | 样本数（应为 1273） |
| `n_match` | 预测与 `answer_idx` 一致的条数 |
| `accuracy` | `n_match / n_scored` |
| `output_csv` | 该组设置的逐条结果 |

参考最优：temp=0.7，ntk=0.4 → **1197 / 1273 ≈ 0.9403**。

```bash
python -c "import pandas as pd; print(pd.read_csv('results/medqa_intent_test1273_eval/summary.csv')[['temperature','ntk_alpha','n_match','accuracy']])"
```

### 可选：论文式下游评测

```bash
lm-eval --model hf \
  --model_args pretrained=./parameters/QF_Struct_300M,trust_remote_code=True \
  --tasks hellaswag,boolq,piqa,arc_easy,winogrande,openbookqa,arc_challenge \
  --device cuda:0 --batch_size 8
```

## 推荐流程（快速复现测试）

```bash
pip install -r requirements.txt
pip install -U modelscope

# 从 ModelScope 拉取 Intent 评测所需权重与测试数据
PACKS="02 04" bash scripts/download_modelscope_assets.sh

CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh

# 可选：阶段二重训（先 PACKS="01 04"）
# bash scripts/run_stage2_intent_sft.sh
```
