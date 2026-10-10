# 数据处理脚本与 Embedding 权重

本目录为 MedQA 数据生成流水线代码与检索用 Embedding 模型。

## 目录

```
pipeline/
├── txt2json.py / vector_store.py / custom_faiss.py / embedding.py / retriever.py
├── generate_question_with_knowledges.py
├── convert_medmcqa_to_medqa_jsonl.py   # MedMCQA → MedQA 风格 jsonl
├── generate_intent_from_retrieval.py   # 检索 JSON → Intent
├── generate_ragmedqa_dataset.py
├── format_json_dataset_for_training_llm.py / *_en.py
├── format_json_dataset_for_eval_from_jsonl.py / *_v2.py
├── add_abstract.py / build_vectorstore.sh / test_code.py / config.py
├── MultiProcessingLLM/          # 其它 LLM 工具（可选）
└── models/
    ├── infgrad/stella-base-en-v2/
    ├── infgrad/stella-base-zh-v2/
    └── stella-large-zh-v2/
```

中间数据写入同级的 `../pipeline_data/`（默认不入库）。流程说明见 [../DATA_PIPELINE_CN.md](../DATA_PIPELINE_CN.md) 与根目录 README「数据集预处理」第 4 步。

## 配置

`config.py` 已将 Embedding 路径指向本目录下 `models/`，一般无需修改。运行脚本时请先 `cd` 到本目录。

Intent 改写脚本的 API 密钥仅通过 `VOLCENGINE_ARK_API_KEY` / `ARK_API_KEY` 或 `--api_key` 提供，**不要**写入代码。

## Embedding 权重

`models/` 下 Stella 权重请从 ModelScope 下载包 `05_embedding_stella.tar`（项目根目录执行 `PACKS="05" bash scripts/download_modelscope_assets.sh`）。详见 [../../MODELSCOPE_ASSETS.md](../../MODELSCOPE_ASSETS.md)。
