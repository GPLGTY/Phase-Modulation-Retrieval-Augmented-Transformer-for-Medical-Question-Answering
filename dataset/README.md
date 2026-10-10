# 数据

大文件不在 Git 中，请从 ModelScope 下载后解压到本目录（或项目根下对应路径）。

仓库：https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering  

```bash
# 在项目根目录
PACKS="04" bash scripts/download_modelscope_assets.sh          # 仅 Intent JSON
PACKS="02 04" bash scripts/download_modelscope_assets.sh       # Intent 评测
PACKS="03a 03b 03c 03d" bash scripts/download_modelscope_assets.sh  # 预训练/SFT 语料
```

详见根目录 [MODELSCOPE_ASSETS.md](../MODELSCOPE_ASSETS.md)、[README_CN.md](../README_CN.md)。

## 解压后应有的文件

| 路径 | 用途 | 条数 | 包 |
|------|------|------|-----|
| `SlimPajama-1B_full.jsonl` | 英文预训练 | — | `03a` |
| `openhermes2_5.jsonl` | 英文通用 SFT | — | `03b` |
| `pretrain_hq_zh.jsonl` | 中文预训练（可选） | — | `03c` |
| `sft_mini_512_zh.jsonl` | 中文 SFT（可选） | — | `03d` |
| `RAG_MedQA_US_train_10178_intent.json` | MedQA Intent SFT 训练 | 10178 | `04` |
| `RAG_MedQA_US_dev_1272_intent.json` | MedQA Intent SFT 验证 | 1272 | `04` |
| `RAG_MedQA_US_test_1273_intent.json` | MedQA Intent 测试 | 1273 | `04` |
| `pipeline/models/` | Stella Embedding | — | `05` |

公开语料备份：https://www.modelscope.cn/datasets/longview/QubitFormerDataset

代码：`lm_en_dataset.py` / `lm_zh_dataset.py` 为训练时数据读取类，请保留。

## 数据处理与生成

| 内容 | 路径 |
|------|------|
| 流程说明 | [DATA_PIPELINE_CN.md](./DATA_PIPELINE_CN.md) |
| 脚本与 Embedding | [pipeline/](./pipeline/)（见 [pipeline/README.md](./pipeline/README.md)） |
| 中间数据（运行生成） | `pipeline_data/`（默认不入库） |
