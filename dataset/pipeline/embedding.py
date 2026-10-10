from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

from config import huggingface_embeddings_config


def load_huggingface_embedding(name, device: str = "cpu") -> HuggingFaceEmbeddings:
    model_path = huggingface_embeddings_config[name]["model_path"]
    model_name: str = huggingface_embeddings_config[name]["model_name"]
    encode_kwargs = huggingface_embeddings_config[name]["encode_kwargs"]

    model_name_or_path = model_path if model_path else model_name
    if model_name.startswith("bge"):
        embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs=encode_kwargs,
            model_kwargs={"device": device},
        )
    else:
        embedding = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs=encode_kwargs,
            model_kwargs={"device": device},
        )

    return embedding
