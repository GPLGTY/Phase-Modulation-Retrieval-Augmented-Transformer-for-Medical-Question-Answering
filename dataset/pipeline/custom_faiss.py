from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document

from typing import Optional, List, Tuple, Dict, Any


class Faiss(FAISS):
    def similarity_search_with_relevance_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        relevance_score_threshold=None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        relevance_score_fn = self._select_relevance_score_fn()
        if relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )

        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, fetch_k=fetch_k, **kwargs
        )

        docs_and_rel_scores = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]

        if relevance_score_threshold != None:
            docs_and_rel_scores = [
                (doc, score)
                for doc, score in docs_and_rel_scores
                if score > relevance_score_threshold
            ]

        return docs_and_rel_scores
