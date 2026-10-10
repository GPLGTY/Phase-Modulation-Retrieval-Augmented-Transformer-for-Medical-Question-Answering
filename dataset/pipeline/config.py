import os

# 本流水线根目录：dataset/pipeline
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库 dataset/ 目录
DATASET_DIR = os.path.dirname(PIPELINE_DIR)
# 仓库根目录
PROJ_TOP_DIR = os.path.dirname(DATASET_DIR)
# 中间数据目录（生成时写入，可不入库）
PIPELINE_DATA_DIR = os.path.join(DATASET_DIR, "pipeline_data")
# Embedding 权重目录
MODELS_DIR = os.path.join(PIPELINE_DIR, "models")

huggingface_embeddings_config = {
    "stella-base-en-v2": {
        "model_name": "infgrad/stella-base-en-v2",
        "model_path": os.path.join(MODELS_DIR, "infgrad", "stella-base-en-v2"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 1024,
    },
    "stella-base-zh-v2": {
        "model_name": "infgrad/stella-base-zh-v2",
        "model_path": os.path.join(MODELS_DIR, "infgrad", "stella-base-zh-v2"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 1024,
    },
    "stella-large-zh-v2": {
        "model_name": "stella-large-zh-v2",
        "model_path": os.path.join(MODELS_DIR, "stella-large-zh-v2"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 1024,
    },
}
