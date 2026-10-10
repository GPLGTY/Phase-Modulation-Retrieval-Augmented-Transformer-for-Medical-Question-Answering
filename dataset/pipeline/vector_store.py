from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from config import huggingface_embeddings_config
from embedding import load_huggingface_embedding
from custom_faiss import Faiss
from multiprocessing import Pool, Lock
from tqdm import tqdm
import gc
import time
import torch
import argparse


def __new_relevance_score_fn(distance: float) -> float:
    # 将余弦相似度从[-1, 1]映射至[0, 1]中
    return 0.5 + 0.5 * distance


def __load_store(store_path: str, embedding_model: HuggingFaceEmbeddings) -> Faiss:
    return Faiss.load_local(
        folder_path=store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True, # 修复读入时报错：The de-serialization relies loading a pickle file. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine.You will need to set `allow_dangerous_deserialization` to `True` to enable deserialization. If you do this, make sure that you trust the source of the data. For example, if you are loading a file that you created, and know that no one else has modified the file, then this is safe to do. Do not set this to `True` if you are loading a file from an untrusted source (e.g., some random site on the internet.).
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        relevance_score_fn=__new_relevance_score_fn,
    )


def load_store_for_one_stage(store_path: str, embedding_model: HuggingFaceEmbeddings) -> Faiss:
    embedding_name = embedding_model.model_name.split("/")[-1]
    store_path = os.path.join(store_path, embedding_name, "0_whole")
    whole_store = __load_store(
        store_path,
        embedding_model=embedding_model,
    )
    return whole_store


def load_store_for_two_stage(store_path: str, 
    embedding_model: HuggingFaceEmbeddings,
):
    embedding_name = embedding_model.model_name.split("/")[-1]
    names_store_path = os.path.join(
        store_path, embedding_name, "1_names"
    )
    abstracts_store_path = os.path.join(
        store_path, embedding_name, "2_abstracts"
    )

    names_store = __load_store(
        store_path=names_store_path,
        embedding_model=embedding_model,
    )

    abstracts_store = __load_store(
        store_path=abstracts_store_path, embedding_model=embedding_model
    )

    names = os.listdir(os.path.join(store_path, embedding_name))
    names.remove("0_whole")
    names.remove("1_names")
    names.remove("2_abstracts")
    stores = {
        name: __load_store(
            os.path.join(store_path, embedding_name, name),
            embedding_model=embedding_model,
        )
        for name in names
    }

    return names_store, abstracts_store, stores


def build_vectorstores_per_name(
    json_dir: str,
    store_top_path: str,
    embedding_model: HuggingFaceEmbeddings,
    tqdm_position: int,
) -> None:
    embedding_name = embedding_model.model_name.split("/")[-1]
    max_len = huggingface_embeddings_config[embedding_name]["max_len"]

    file_names = os.listdir(json_dir)

    store_names = []
    abstracts = []
    abstract_metadatas = []

    deduplicated_texts = []
    for file_name in tqdm(
        file_names,
        desc=f"{embedding_name} build_vectorstores_per_name",
        position=tqdm_position * 2,
    ):
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r") as f:
            data = json.load(fp=f)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "。", "！", "？", "；", "，", ""],
            chunk_size=max_len,
            chunk_overlap=0,
            length_function=len,
        )
        texts_process = []
        metadatas = []

        for id, t in data["knowledges"].items():
            if t in deduplicated_texts:
                continue
            else:
                deduplicated_texts.append(t)
                if len(t) > max_len:
                    splits = text_splitter.split_text(text=t)
                    texts_process.extend(splits)
                    metadatas.extend(
                        [
                            {
                                "file_name": file_name.split(".")[0],
                                "id": id,
                                "knowledge": t,
                            }
                            for _ in splits
                        ]
                    )
                else:
                    texts_process.append(t)
                    metadatas.append(
                        {"file_name": file_name.split(".")[0], "id": id, "knowledge": t}
                    )
        if len(texts_process) > 0:
            store_name = file_name.split(".")[0]
            store_names.append(file_name.split(".")[0])

            abstracts.append(data["abstract"])
            abstract_metadatas.append({"file_name": file_name.split(".")[0]})

            txt_vecstore = Faiss.from_texts(
                texts=texts_process,
                embedding=embedding_model,
                # 距离度量使用余弦相似度
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                relevance_score_fn=__new_relevance_score_fn,
                metadatas=metadatas,
            )
            store_path = os.path.join(
                store_top_path,
                f"{embedding_name}/{store_name}",
            )
            txt_vecstore.save_local(store_path)

    names_vecstore = Faiss.from_texts(
        texts=store_names,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    names_vecstore.save_local(os.path.join(store_top_path, f"{embedding_name}/1_names"))

    abstracts_vecstore = Faiss.from_texts(
        texts=abstracts,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        metadatas=abstract_metadatas,
    )
    abstracts_vecstore.save_local(
        os.path.join(store_top_path, f"{embedding_name}/2_abstracts")
    )


def build_whole_vectorstore(
    json_dir: str,
    store_top_path: str,
    embedding_model: HuggingFaceEmbeddings,
    tqdm_position: int,
    rebuilding_knowledge=False

) -> None:
    embedding_name = embedding_model.model_name.split("/")[-1]
    max_len = huggingface_embeddings_config[embedding_name]["max_len"]

    file_names = os.listdir(json_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "。", "！", "？", "；", "，", ". ", "?", "!", ""],
        chunk_size=max_len,
        chunk_overlap=0,
        length_function=len,
    )
    whole_texts = []
    metadatas = []
    deduplicated_texts = []
    for file_name in tqdm(
        file_names,
        desc=f"{embedding_name} build_whole_vectorstore",
        position=tqdm_position * 2 + 1,
    ):
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r") as f:
            data = json.load(fp=f)

        # 陈艺荣修改知识关系
        i = 0
        for id, t in data["knowledges"].items():
            if t in deduplicated_texts:
                continue
            else:
                deduplicated_texts.append(t)
                if len(t) > max_len:
                    splits = text_splitter.split_text(text=t)
                    whole_texts.extend(splits)

                    if rebuilding_knowledge:

                        for split_text in splits:
                            # 陈艺荣修改
                            metadatas.append(
                                {
                                    "file_name": file_name.split(".")[0],
                                    "id": i,
                                    "knowledge": split_text,
                                }
                            )
                            i = i + 1
                    else:
                        metadatas.extend(
                            [
                                {
                                    "file_name": file_name.split(".")[0],
                                    "id": id,
                                    "knowledge": t,
                                }
                                for _ in splits
                            ]
                        )
                else:
                    whole_texts.append(t)
                    if rebuilding_knowledge:
                        metadatas.append(
                            {"file_name": file_name.split(".")[0], "id": id, "knowledge": t}
                        )
                        i = i + 1
                    else:
                        metadatas.append(
                            {"file_name": file_name.split(".")[0], "id": id, "knowledge": t}
                        )

    print("开始构建whole_vecstore")
    whole_vecstore = Faiss.from_texts(
        texts=whole_texts,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        relevance_score_fn=__new_relevance_score_fn,
        metadatas=metadatas,
    )
    whole_vecstore.save_local(os.path.join(store_top_path, f"{embedding_name}/0_whole"))


def process(
    json_dir: str,
    store_top_path: str,
    embedding_model_name: str,
    tqdm_position: int,
    device: int,
    rebuilding_knowledge=False
):
    embedding_model = load_huggingface_embedding(embedding_model_name, device=device)

    build_whole_vectorstore(
        json_dir=json_dir,
        store_top_path=store_top_path,
        embedding_model=embedding_model,
        tqdm_position=tqdm_position,
        rebuilding_knowledge=rebuilding_knowledge,
    )

    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--input_json_dir", type=str, default='../pipeline_data/textbooks/zh_sentence_json', help="json格式的知识文档的目录")
    parser.add_argument("-oj", "--store_top_path", type=str, default='../pipeline_data/vector_stores/zh_sentence', help="张量保存的目录")
    parser.add_argument("-d", "--device_ids", type=str, default='0,1,2,3,4,5,6,7', help="字符串形式，允许使用的device_id，以,分隔")
    parser.add_argument("-rk", '--rebuilding_knowledge', action='store_true', help="设置为True时，对输入的知识进行拆分时会重新编号，每个拆分的文本作为一个知识")


    args = parser.parse_args()

    input_json_dir = args.input_json_dir
    store_top_path = args.store_top_path
    device_ids = args.device_ids
    device_ids = [int(i) for i in device_ids.split(',')] # '0,1,2,3,4,5,6,7' --> [0, 1, 2, 3, 4, 5, 6, 7]

    t = time.time()

    if not os.path.exists(store_top_path):
        os.makedirs(store_top_path)

    embedding_model_list = list(huggingface_embeddings_config.keys())

    process_num = len(embedding_model_list)

    # 自动化构建device_list
    device_list = []
    device_num = len(device_ids)
    for i in range(process_num):
        device_list.append(device_ids[i%device_num])

    pool = Pool(processes=process_num, initializer=tqdm.set_lock, initargs=(Lock(),))

    pool.starmap(
        process,
        list(
            zip(
                [input_json_dir] * process_num,
                [store_top_path] * process_num,
                embedding_model_list,
                list(range(process_num)),
                device_list,
            )
        ),
    )

    pool.close()
    pool.join()

    print(f"程序用时：", time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
