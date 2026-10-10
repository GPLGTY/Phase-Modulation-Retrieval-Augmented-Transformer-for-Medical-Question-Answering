import json
import os

# 示例：将 path 改为本地检索结果 JSON 后运行
_here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    _here,
    "..",
    "pipeline_data",
    "questions_with_knowledge",
    "US",
    "test",
    "retrive_use_question",
    "stella-base-en-v2",
    "test.json",
)

if __name__ == "__main__":
    if not os.path.isfile(data_path):
        raise SystemExit(f"文件不存在，请先完成检索或修改 data_path: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(type(data), len(data) if hasattr(data, "__len__") else data)
