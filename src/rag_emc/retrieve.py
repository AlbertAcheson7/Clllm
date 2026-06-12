import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from BCEmbedding import EmbeddingModel, RerankerModel
import chromadb
import numpy as np
from typing import Any, Mapping, Optional

HIERARCHICAL_COLLECTION = "mesh_hierarchical"
SYNONYM_COLLECTION = "mesh_synonym"
DEFINITION_COLLECTION = "mesh_definition"

class Retriever:
    """
    处理从三个不同知识源进行的两阶段检索过程:
    1. 从向量数据库进行粗粒度检索。
    2. 对结果进行细粒度重排序。
    """
    def __init__(self, embedding_model_path, reranker_model_path, chroma_db_path, device="cuda:0"):
        self.embedding_model = EmbeddingModel(model_name_or_path=embedding_model_path, device=device)
        self.reranker_model = RerankerModel(model_name_or_path=reranker_model_path, device=device)
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection_names = {
            "hierarchical": HIERARCHICAL_COLLECTION,
            "synonym": SYNONYM_COLLECTION,
            "definition": DEFINITION_COLLECTION
        }
        self.collections = {
            key: chroma_client.get_collection(name=name)
            for key, name in self.collection_names.items()
        }

    def retrieve(self, mention: str, top_n_coarse: int, prompt_templates: Optional[Mapping[str, str]] = None):
        results = {}
        for key, collection in self.collections.items():
            collection_name = self.collection_names[key]
            if prompt_templates and key in prompt_templates:
                query_text = prompt_templates[key].format(mention=mention)
            else:
                query_text = mention
            query_embedding = self.embedding_model.encode([query_text])[0]
            query_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_n_coarse
            )
            doc_list = query_results.get('documents')
            meta_list = query_results.get('metadatas')
            if not doc_list or not meta_list or not doc_list[0]:
                results[key] = {
                    "document": None,
                    "metadata": None,
                    "score": -1.0
                }
                continue
            documents = doc_list[0]
            metadatas = meta_list[0]
            reranker_pairs = [(mention, doc) for doc in documents]
            scores = self.reranker_model.compute_score(reranker_pairs)
            best_doc_index = np.argmax(scores)
            best_score = scores[best_doc_index]
            best_document = documents[best_doc_index]
            best_metadata = metadatas[best_doc_index]
            results[key] = {
                "document": best_document,
                "metadata": best_metadata,
                "score": best_score
            }
        return results
