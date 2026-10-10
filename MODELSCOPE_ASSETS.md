# ModelScope 权重与数据

大文件已托管在公开模型库：

**https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering**

仓库 ID：`GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`

GitHub / 本仓库只放代码；权重、语料、Embedding 用下面方式拉取后，路径与训练/评测脚本默认值一致。

## 一键下载（推荐）

在**本项目根目录**执行：

```bash
pip install -U modelscope   # 若尚未安装

# 全量（约 17G）
bash scripts/download_modelscope_assets.sh

# 仅 Intent 测试集评测（约 1.7G）：Intent 权重 + MedQA Intent JSON
PACKS="02 04" bash scripts/download_modelscope_assets.sh

# Intent SFT：阶段一权重 + Intent 数据
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 本地重跑检索流水线时再加 Embedding
PACKS="01 04 05" bash scripts/download_modelscope_assets.sh
```

脚本会下载指定 `.tar` 并解压到项目内正确相对路径。

## 远程文件一览

| 文件 | 约大小 | 解压后位置 | 用途 |
|------|--------|------------|------|
| `01_parameters.tar` | 6.4G | `parameters/` | QF/MiniMind HF + `sft_QF_en_1024.pth` |
| `02_intent_sft_checkpoints.tar` | 1.5G | `out/run_sft_intent/` | Intent SFT（**评测用 `*_best.pth`**） |
| `03a_corpus_SlimPajama-1B_full.tar` | 3.9G | `dataset/SlimPajama-1B_full.jsonl` | 英文预训练 |
| `03b_corpus_openhermes2_5.tar` | 1.6G | `dataset/openhermes2_5.jsonl` | 英文通用 SFT |
| `03c_corpus_pretrain_hq_zh.tar` | 1.6G | `dataset/pretrain_hq_zh.jsonl` | 中文预训练（可选） |
| `03d_corpus_sft_mini_512_zh.tar` | 1.2G | `dataset/sft_mini_512_zh.jsonl` | 中文 SFT（可选） |
| `04_medqa_intent_json.tar` | 213M | `dataset/RAG_MedQA_US_*_intent.json` | Intent 训练/验证/测试 |
| `05_embedding_stella.tar` | 450M | `dataset/pipeline/models/` | Stella Embedding |
| `SHA256SUMS.txt` / `MANIFEST.md` | 小 | — | 校验与说明 |

## 按场景选用

| 目标 | 建议 `PACKS` |
|------|----------------|
| 复现 Intent 测试准确率 | `02 04` |
| 在阶段一权重上跑 Intent SFT | `01 04` |
| 预训练 / 通用 SFT 从头训 | 再加 `03a` / `03b`（中文再加 `03c`/`03d`） |
| 从原始题库重跑检索流水线 | 再加 `05` |

## 手动下载

1. 打开上述 ModelScope 页面，下载需要的 `.tar`
2. 放到任意目录后，在**本项目根目录**执行：`tar -xf /path/to/xxx.tar`
3. 勿解压到额外子目录；包内已含 `parameters/`、`dataset/` 等相对路径

校验（可选）：

```bash
cd .modelscope_packs   # 或你保存 tar 的目录
sha256sum -c SHA256SUMS.txt
```
