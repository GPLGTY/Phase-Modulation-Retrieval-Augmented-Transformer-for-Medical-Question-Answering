import time
import json
from pydantic.dataclasses import dataclass
from pydantic import Field
from langchain.embeddings import HuggingFaceEmbeddings
from vector_store import load_store_for_one_stage, load_store_for_two_stage


@dataclass
class SearchResult:
    time: float = 0.0

    knowledges: list[str] = Field(default_factory=list)
    knowledge_scores: list[float] = Field(default_factory=list)
    knowledge_names: list[str] = Field(default_factory=list)
    knowledge_ids: list[int] = Field(default_factory=list)

    knowledges_for_llm: str = ""
    names_or_abstracts: list[str] = Field(default_factory=list)
    names_or_abstracts_scores: list[float] = Field(default_factory=list)


class OneStageRetriever:
    def __init__(self, 
                 store_path: str,
                 embedding_model: HuggingFaceEmbeddings):
        self.whole_store = load_store_for_one_stage(store_path=store_path, embedding_model=embedding_model)

    def search(
        self, query: str, topk_knowledge: int, knowledge_threshold=None
    ) -> SearchResult:
        t0 = time.process_time()

        # 搜索结果按照分数从大到小排序
        query_emb = self.whole_store._embed_query(query)
        docs_with_scores = (
            self.whole_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_knowledge,
                relevance_score_threshold=knowledge_threshold,
            )
        )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            # 输入到llm中的knowledges按照分数从小到大排序
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )


class TwoStageRetriever:
    def __init__(self, 
                 store_path: str,
                 embedding_model: HuggingFaceEmbeddings):
        self.names_store, self.abstracts_store, self.stores = load_store_for_two_stage(
            store_path=store_path,
            embedding_model=embedding_model
        )

    def search_by_name(
        self,
        query: str,
        topk_name: int,
        topk_knowledge: int,
        knowledge_threshold=None,
    ) -> SearchResult:
        t0 = time.process_time()

        # 第一阶段，搜索结果按照分数从大到小排序
        query_emb = self.names_store._embed_query(query)
        names_with_scores = (
            self.names_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_name,
            )
        )

        # 第二阶段
        docs_with_scores = []
        for name, _ in names_with_scores:
            docs_with_scores.extend(
                self.stores[
                    name.page_content
                ].similarity_search_with_relevance_score_by_vector(
                    embedding=query_emb,
                    k=topk_knowledge,
                    relevance_score_threshold=knowledge_threshold,
                )
            )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        # 搜索结果按照分数从大到小排序
        docs_with_scores = sorted(docs_with_scores, key=lambda x: -x[1])[
            0:topk_knowledge
        ]

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            names_or_abstracts=[d.page_content for d, _ in names_with_scores],
            names_or_abstracts_scores=[s for _, s in names_with_scores],
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )

    def search_by_abstract(
        self,
        query: str,
        topk_abstract: int,
        topk_knowledge: int,
        knowledge_threshold=None,
    ) -> SearchResult:
        t0 = time.process_time()

        # 第一阶段，搜索结果按照分数从大到小排序
        query_emb = self.abstracts_store._embed_query(query)
        abstracts_with_scores = (
            self.abstracts_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_abstract,
            )
        )

        # 第二阶段
        docs_with_scores = []
        for abstract, _ in abstracts_with_scores:
            docs_with_scores.extend(
                self.stores[
                    abstract.metadata["file_name"]
                ].similarity_search_with_relevance_score_by_vector(
                    embedding=query_emb,
                    k=topk_knowledge,
                    relevance_score_threshold=knowledge_threshold,
                )
            )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        # 搜索结果按照分数从大到小排序
        docs_with_scores = sorted(docs_with_scores, key=lambda x: -x[1])[
            0:topk_knowledge
        ]

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            names_or_abstracts=[
                f"{d.metadata['file_name']}\n" + d.page_content
                for d, _ in abstracts_with_scores
            ],
            names_or_abstracts_scores=[s for _, s in abstracts_with_scores],
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )




def load_jsonl_data(data_path):
    '''
    功能：从data_path中读取jsonl文件，并且返回列表
    '''

    # 读取所有行并解析为JSON对象的列表
    with open(data_path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]

    return data_list


def convert_questions_dict_example_to_new_example(questions_dict):
    '''
    questions_dict = {'question': '经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（\u3000\u3000）。', 
                      'options': {'A': '2小时', 'B': '4小时内', 'C': '8小时内', 'D': '12小时内', 'E': '24小时内'}, 
                      'answer': '24小时内', 
                      'meta_info': '卫生法规', 
                      'answer_idx': 'E'}
    
    '''

    options_str = '\n'.join([f'{k}. {v}' for k, v in questions_dict["options"].items()])

    answer_with_idx = questions_dict['answer_idx'] + '. ' + questions_dict['answer']

    question_with_options = questions_dict['question'] + '\n' + options_str

    return {
        'question_with_options': question_with_options,
        'answer_with_idx': answer_with_idx,
        'question': questions_dict['question'],
        'options': questions_dict['options'],
        'answer': questions_dict['answer'],
        'meta_info': questions_dict['meta_info'], 
        'answer_idx': questions_dict['answer_idx']
    }



if __name__ == "__main__":

    import argparse
    from config import huggingface_embeddings_config
    from embedding import load_huggingface_embedding
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_jsonl_path", type=str, default='../pipeline_data/questions/Mainland/train.jsonl', help="json格式的知识文档的目录")
    parser.add_argument("-n", "--embedding_model_name", type=str, default='stella-base-zh-v2', help="字符串形式，输入的embedding模型的名称")
    parser.add_argument("-e", "--embedding_model_path", type=str, default='', help="embedding 模型路径；为空则使用 config 中对应名称")
    parser.add_argument("-v", "--store_path", type=str, default='../pipeline_data/vector_stores/zh_paragraph', help="张量数据库保存的目录，其子目录类似于：stella-base-zh-v2/0_whole")
    parser.add_argument("-o", "--output_json_path", type=str, default='../pipeline_data/questions_with_knowledge/zh_paragraph', help="张量保存的目录")
    
    args = parser.parse_args()

    input_jsonl_path = args.input_jsonl_path
    embedding_model_name = args.embedding_model_name
    embedding_model_path = args.embedding_model_path
    store_path = args.store_path
    output_json_path = args.output_json_path

    data_list = load_jsonl_data(input_jsonl_path)

    new_data_list = [convert_questions_dict_example_to_new_example(questions_dict) for questions_dict in data_list]

    # 进行检索
    embedding_model = load_huggingface_embedding(name=embedding_model_name, device=device)
    one_stage_retriever = OneStageRetriever(embedding_model=embedding_model)
    two_stage_retriever = TwoStageRetriever(embedding_model=embedding_model)









