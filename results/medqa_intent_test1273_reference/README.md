# MedQA Intent 测试参考结果

目录内为参考评测 CSV 与 `summary.csv`。

复现前先从 ModelScope 下载 Intent 权重与测试数据：

```bash
PACKS="02 04" bash scripts/download_modelscope_assets.sh

CHECKPOINT=out/run_sft_intent/sft_intent_QF_en_1024_best.pth \
  bash scripts/eval_medqa_test.sh
```

默认评测设置：`temperature=0.7`，`ntk_alpha=0.4`。
