# MedQA 数据处理与生成流程（草案）

> **状态**：草案。步骤与命令已按本仓库口径整理，便于后续取舍与增补。  
> **工作目录约定**：下文命令请在 `dataset/pipeline/` 下执行（脚本与 Embedding 权重已迁入该目录）。  
> **已附带成品**：当前仓库已提供下游 Intent 数据，完整重跑流水线前可先使用成品文件。

---

## 0. 流水线概览与成品对应关系

```
MedQA 原始题库 + 教材
        │
        ▼
 ① 教材 txt → json 分块
        │
        ▼
 ② json → 向量库
        │
        ▼
 ③ 题目检索增强（题干 / 题干+选项）
        │
        ▼
 ④ generate_intent_from_retrieval.py
     （检索 JSON → Intent 样本）
```

| 本仓库已附带文件 | 用途 |
|------------------|------|
| `RAG_MedQA_US_train_10178_intent.json` | Intent SFT 训练（10178） |
| `RAG_MedQA_US_dev_1272_intent.json` | Intent SFT 验证（1272） |
| `RAG_MedQA_US_test_1273_intent.json` | Intent 测试（1273） |

步骤 ④ 脚本：`dataset/pipeline/generate_intent_from_retrieval.py`。若已附带上表三份 Intent 文件，可跳过 ①–④。

建议的中间数据目录（生成时创建，可不入库）：

```
dataset/pipeline_data/
├── textbooks/                 # en / zh_paragraph / zh_sentence 及对应 *_json
├── questions/                 # US / Mainland 的 train|dev|test.jsonl
├── vector_stores/             # 向量库
└── questions_with_knowledge/  # 检索增强后的题目与解析
```

---

## 1. 原始数据

- MedQA 官方仓库：https://github.com/jind11/MedQA  
- 数据包下载：https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing  
- 中文介绍：https://zhuanlan.zhihu.com/p/679590312  

下载后将题目与教材整理到 `dataset/pipeline_data/questions/`、`dataset/pipeline_data/textbooks/`（具体子目录名与官方包结构对齐即可）。

---

## 2. 预处理步骤

### 2.1 教材 txt → json 分块

```bash
cd dataset/pipeline

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' \
  --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' \
  --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/zh_sentence' \
  --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' \
  --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/en' \
  --output_json_dir '../pipeline_data/textbooks/en_json' \
  --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

### 2.2 json → 向量库

```bash
cd dataset/pipeline

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' \
  --store_top_path '../pipeline_data/vector_stores/zh_sentence'

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' \
  --store_top_path '../pipeline_data/vector_stores/zh_paragraph'

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/en_json' \
  --store_top_path '../pipeline_data/vector_stores/en'
```

运行前在 `dataset/pipeline/config.py`（或等价配置）中将 Embedding 的 `model_path` 设为本机模型路径。

### 2.3 题目检索增强

统一使用 `--query_key_name="question_with_options"`（与步骤 2.4 对齐）。勿先跑 `question` 再跑 `question_with_options` 到同一输出路径，以免覆盖。

参数说明：

- `--device_ids`：GPU 列表；单卡用 `"0"`，并设 `--num_process=1`。  
- `--topk_knowledge` / `--knowledge_threshold`：召回条数与相似度阈值。  
- 产物路径：`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

#### 2.3.1 MedQA 英文 US（Intent 主路线）

```bash
cd dataset/pipeline

python generate_question_with_knowledges.py \
  --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
  --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
  --embedding_model_name="stella-base-en-v2" \
  --store_path="../pipeline_data/vector_stores/en" \
  --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
  --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
  --query_key_name="question_with_options"

# 将 train → dev / test，路径同步改为 US/dev、US/test
```

#### 2.3.2 MedQA 中文 Mainland（可选）

同上，改用 `Mainland/*.jsonl`、`stella-base-zh-v2`、`vector_stores/zh_paragraph`。

#### 2.3.3 MedMCQA（先转换，再检索）

原始为 `opa/opb/opc/opd` + `cop`，先转为 MedQA 风格 jsonl：

```bash
cd dataset/pipeline

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

# 验证集：MedMCQA/dev.jsonl → questions_with_knowledge/MedMCQA/dev
```

### 2.4 检索 JSON → Intent

脚本：`dataset/pipeline/generate_intent_from_retrieval.py`。`--in_json` 须为 2.3 的 `question_with_options` 产物。

```bash
# 密钥仅通过环境变量或 CLI，勿写入代码
export VOLCENGINE_ARK_API_KEY='你的密钥'   # 或 ARK_API_KEY

cd dataset/pipeline

# MedQA US
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedQA_US_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8

# MedMCQA（承接 2.3.3）
python generate_intent_from_retrieval.py \
  --in_json ../pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
  --out_json ../RAG_MedMCQA_train_intent.json \
  --model doubao-seed-2-0-mini-260428 \
  --workers 8
# 可选：--with_rag  --offset 0 --n 100  --base_url ... --api_key ...
```

输出可用于 Intent SFT；也可从 ModelScope 下载 `04_medqa_intent_json.tar`（`PACKS="04"`）跳过生成。完整命令见根目录 README。

---

## 3. 环境与模型（数据处理相关）

| 用途 | 说明 |
|------|------|
| Python / PyTorch | 与仓库根目录 `requirements.txt` 对齐；检索与向量化另需 Embedding 依赖 |
| Embedding | 中文示例 `stella-base-zh-v2`，英文示例 `stella-base-en-v2`（路径在 config 中配置） |
| LLM API | 步骤 2.4 需要 OpenAI 兼容接口；密钥用环境变量 / `--api_key` |
| 模型下载 | 可用 https://hf-mirror.com/ 等镜像加速 |

---

## 4. 已迁入内容

| 类别 | 路径 |
|------|------|
| 流水线脚本 | `dataset/pipeline/*.py`、`build_vectorstore.sh` |
| MedMCQA 转换 | `dataset/pipeline/convert_medmcqa_to_medqa_jsonl.py` |
| Intent 改写 | `dataset/pipeline/generate_intent_from_retrieval.py` |
| Embedding 权重 | `dataset/pipeline/models/infgrad/stella-base-{en,zh}-v2/`、`models/stella-large-zh-v2/` |
| 路径配置 | `dataset/pipeline/config.py`（自动指向本目录 models） |

## 5. 待定事项

- [ ] 补充 `langchain` / `faiss` / `openai` 等流水线依赖说明（若与根目录 `requirements.txt` 不同）  


