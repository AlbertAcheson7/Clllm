# *******************  RAG 相关 *******************

RAG_MENTION_PATH = "/home/zhangzy/CLllm/src/dictionary/output/bc5cdr_train_da_mention_dependencyTree.json"

MESH_PATH = "/home/zhangzy/CLllm/data/data_pre/dict_pred/MeSH2017_desc2017_parsed_with_names.json"
BC5CDR_DIR_PATH = "/home/zhangzy/CLllm/data/BC5CDR"
NCBI_DIR_PATH = "/home/zhangzy/CLllm/data/NCBI"
DICTIONARY_OUTPUT_DIR_PATH = "/home/zhangzy/CLllm/src/dictionary/output"



# RAG 生成模型model_name 
MODEL_NAME = "gpt2"
# RAG 文本向量化模型embedder_name
EMBEDDER_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
# RAG 索引路径
INDEX_PATH = '/home/zhangzy/CLllm/src/rag/index/knowledge_base_mesh_IndexFlatIP.faiss'







# 定义型prompt
RETRIEVE_PROMPT = ["What is {mention}?", 
    "Define {mention}", 
    "Explain {mention} in the context of {context}"]












########################################################
# 非rag定义的config

# 后续可能能用得到。
# 上下位关系 prompt（Hierarchy-style prompts）
RETRIEVE_HIERARCHY_PROMPTS = [
    "Which category does {mention} belong to?",
    "What broader category includes {mention}?",
    "Is {mention} a type of disease or chemical?"
]

# 相似性 prompt（Synonym-style prompts）
RETRIEVE_SYNONYM_PROMPTS = [
    "What are synonyms of {mention}?",
    "Is there another term for {mention}?",
    "Does {mention} refer to the same concept as any known medical term?"
]

# 上下文式 prompt（Context-aware prompts）
RETRIEVE_CONTEXT_PROMPTS = [
    "In the sentence: \"{context}\", what does \"{mention}\" refer to?",
    "Given the context \"{context}\", explain the meaning of \"{mention}\".",
    "In the phrase from a clinical note: \"{context}\", is \"{mention}\" a medical term?"
]



PROMPT_TEMPLATES = {
    "default": '''Context (Knowledge Base):
{context}

Task: Extract named entities from the following text.
Entity types to identify: {entity_types}

Text: {text}

Instructions:
1. Identify all entities in the text
2. Classify each entity by type
3. Provide explanation based on the context

Format your response as JSON:
{{"entities": [{{"name": "entity_name", "type": "entity_type", "explanation": "why this classification"}}]}}

Response:''',
    # 你可以在这里添加更多模板
}
