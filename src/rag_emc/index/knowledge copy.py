import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from BCEmbedding import EmbeddingModel
import chromadb

# 数据路径
MESH_JSON_PATH = "/home/zhangzy/CLllm/data/data_pre/dict_pred/MeSH2017_desc2017_parsed_with_names.json"

# 初始化BCEmbedding模型
embedding_model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")

# 初始化Chroma向量数据库
chroma_client = chromadb.PersistentClient(path="/home/zhangzy/CLllm/src/rag_emc/index/chroma_db")
collection_hierarchical = chroma_client.get_or_create_collection(name="mesh_hierarchical")
collection_synonym = chroma_client.get_or_create_collection(name="mesh_synonym")
collection_definition = chroma_client.get_or_create_collection(name="mesh_definition")

def get_entity_type(tree_numbers):
    if not tree_numbers:
        return "chemical"  # 默认化学
    if tree_numbers[0].startswith("C"):
        return "disease"
    return "chemical"

def generate_texts(entry):
    entity_name = entry.get("entity_name", "")
    entity_definition = entry.get("entity_definition", "")
    synonyms = entry.get("synonyms", [])
    tree_numbers = entry.get("tree_numbers", [])
    tree_numbers_name = entry.get("tree_numbers_name", [])
    entity_type = get_entity_type(tree_numbers)
    type_en = "disease" if entity_type == "disease" else "chemical"
    
    # 检查 tree_numbers_name 是否为空
    category_name = tree_numbers_name[0] if tree_numbers_name else "unknown category"
    
    # 上下级-信息
    hierarchical_texts = []
    hierarchical_texts.append(f'"{entity_name}" is a type of "{category_name}", which belongs to a {type_en} entity.')
    hierarchical_texts.append(f'{type_en.capitalize()} entity "{entity_name}" belongs to the category of "{category_name}".')
    
    # 同义词-信息
    synonym_texts = []
    for synonym in synonyms:
        synonym_texts.append(f'"{synonym}", a synonym of "{entity_name}", refers to the same {type_en} entity.')
        synonym_texts.append(f'"{synonym}" is another name for "{entity_name}", which is categorized as a {type_en}.')

    # 定义-信息
    definition_texts = []
    definition_texts.append(f'"{entity_name}" is described as "{entity_definition}", belongs to the {type_en} class.')
    definition_texts.append(f'"{entity_name}", classified as a {type_en} entity, is defined as "{entity_definition}".')
    
    return hierarchical_texts, synonym_texts, definition_texts

def main():
    with open(MESH_JSON_PATH, 'r', encoding='utf-8') as f:
        mesh_data = json.load(f)

    collections = {
        "hierarchical": collection_hierarchical,
        "synonym": collection_synonym,
        "definition": collection_definition
    }

    for entry in mesh_data:
        hierarchical_texts, synonym_texts, definition_texts = generate_texts(entry)
        
        texts_map = {
            "hierarchical": hierarchical_texts,
            "synonym": synonym_texts,
            "definition": definition_texts
        }

        mesh_ui = entry.get("entity_id", "")
        entity_name = entry.get("entity_name", "")
        entity_type = get_entity_type(entry.get("tree_numbers", []))
        synonyms = entry.get("synonyms", [])

        for text_type, texts in texts_map.items():
            texts = [t for t in texts if t.strip()]
            if not texts:
                continue

            embeddings = embedding_model.encode(texts)
            collection = collections[text_type]
            
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                collection.add(
                    documents=[text],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        "entity_name": entity_name,
                        "type": entity_type,
                        "Mesh_ID": mesh_ui,# 就是entity_id
                        "synonyms": str(synonyms)
                    }],
                    ids=[f'{mesh_ui}:{text_type}:{i+1}']
                )

"""

embeddings = embedding_model.encode(texts)
collection = collections[text_type]

for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    metadata = {
        "entity_name": entity_name,
        "type": entity_type,
        "Mesh_ID": mesh_ui,  # 就是 entity_id
        "synonyms": str(synonyms)
    }

    # 如果当前是定义，检查是否为空
    if text_type == "definition":
        if text.strip() == "":
            metadata["has_definition"] = False
            metadata["definition_status"] = "Missing"
        else:
            metadata["has_definition"] = True
            metadata["definition_status"] = "Present"

    collection.add(
        documents=[text],
        embeddings=[embedding.tolist()],
        metadatas=[metadata],
        ids=[f'{mesh_ui}:{text_type}:{i+1}']
    )

"""

if __name__ == "__main__":
    main()