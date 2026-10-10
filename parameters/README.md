# 权重目录

大文件不在 Git 中。请从 ModelScope 下载后解压到本目录（及 `../out/run_sft_intent/`）：

```bash
# 在项目根目录
PACKS="01" bash scripts/download_modelscope_assets.sh   # → parameters/
PACKS="02" bash scripts/download_modelscope_assets.sh   # → out/run_sft_intent/
```

仓库：https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering  

| 路径 | 说明 | 包 |
|------|------|-----|
| `QF_Base_300M/pytorch_model.bin` | QF Base HF 权重 | `01` |
| `QF_Struct_300M/pytorch_model.bin` | QF Struct HF 权重 | `01` |
| `MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base HF 权重 | `01` |
| `MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct HF 权重 | `01` |
| `sft_QF_en_1024.pth` | 英文通用 SFT（阶段一） | `01` |

Intent SFT 权重在 `../out/run_sft_intent/`（包 `02`）：

| 路径 | 说明 |
|------|------|
| `sft_intent_QF_en_1024_best.pth` | 验证集最优（推荐用于测试） |
| `sft_intent_QF_en_1024.pth` | 最后一轮 |

也可按 README 自行训练生成。HF 目录含 config、tokenizer、模型代码；可用 `tools/convert_model.py` 在 `.pth` 与 HF 格式之间转换。
