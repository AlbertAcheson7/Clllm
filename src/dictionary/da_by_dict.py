from flashtext import KeywordProcessor
import json
import os

# 用Mesh对BC5CDR和NCBI数据集进行标注，并保存为json文件。

# ===========================1. 加载实体+ 语料词典  ===========================
def load_Mesh_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as file:
        entity_entries = json.load(file)

    Mesh_diseases_chemicals_dict = {} # 仅保留'chemicals' 和 'diseases' 两种实体类型
    for entry in entity_entries:
        entity_id = entry['entity_id']
        tree_numbers = entry.get('tree_numbers', [])
        entity_type = None
        for tree in tree_numbers:
            if tree.startswith('D'):
                entity_type = 'chemicals'  # TODO 
                break
            elif tree.startswith('C'):
                entity_type = 'diseases'
                break
        if entity_type in ['chemicals','diseases']: # 只保留'chemicals','diseases' 这两种
            for synonym in entry['synonyms']:
                Mesh_diseases_chemicals_dict[synonym.lower()] = (entity_id, entity_type)  #  synonym 是 [name:(entity_id, entity_type)]
    return Mesh_diseases_chemicals_dict

def get_corpus_sentences(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = []
    for item in data:
        sentences.append({'sentence_id': item.get('sentence_id'), 'text': item['text']})  
    return sentences
  

# ===========================2. match =========================================
def init_keyword_processor(dict_dict):
    kp = KeywordProcessor(case_sensitive=False)
    for synonym in dict_dict.keys(): 
        kp.add_keyword(synonym)
    return kp

# 匹配实体
def match_entities(sentences, kp, mesh_dict):
    matched_entities = []
    print(f"Matching entities in {len(sentences)} sentences...")
    for sentence_data in sentences:
        sentence = sentence_data['text']
        sentence_id = sentence_data.get('sentence_id')
        entities = kp.extract_keywords(sentence, span_info=True)  # [('apomorphine', 76, 87), ('induced hypothermia', 103, 122)]
        entity_list = []
        for mention, start, end in entities:  #  
            mention_lower = mention.lower()
            entity_id, entity_type = mesh_dict.get(mention_lower, (None, None))
            entity_list.append((mention, start, end, entity_type, entity_id))
        
        matched_entities.append({'sentence_id': sentence_id, 'sentence': sentence, 'entities': entity_list})
    return matched_entities

# 保存(sentence, entities)
def save_matched_entities(matched_entities, matched_triple_path):
    # 自动创建目录
    print(matched_triple_path)
    os.makedirs(os.path.dirname(matched_triple_path), exist_ok=True)
    # 保存为json list，每个元素是{"sentence":..., "entities":...}
    with open(matched_triple_path, 'w', encoding='utf-8') as file:
        json.dump(matched_entities, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
   
    MESH_PATH = "/home/zhangzy/CLllm/data/data_pre/dict_pred/MeSH2017_desc2017_parsed_with_names.json"
    
    BC5CDR_DIR_PATH = "/home/zhangzy/CLllm/data/data_pre/data_pred/BC5CDR"
    NCBI_DIR_PATH = "/home/zhangzy/CLllm/data/data_pre/data_pred/NCBI"
    DICTIONARY_OUTPUT_DIR_PATH = "/home/zhangzy/CLllm/src/dictionary/output"

    mesh_dict_path =  MESH_PATH  
    bc5cdr_corpus_path = BC5CDR_DIR_PATH + "/merged_bc5cdr_train_data.json" 
    nbci_corpus_path = NCBI_DIR_PATH + "/merged_ncbi_train_data.json"
    
    bc5cdr_dictionary_output_dir_path = DICTIONARY_OUTPUT_DIR_PATH + "/BC5CDR/bc5cdr_train_da.json"
    nbci_dictionary_output_dir_path = DICTIONARY_OUTPUT_DIR_PATH + "/NCBI/ncbi_train_da.json"

    # 字典标注并保存
    Mesh_dict = load_Mesh_dict(mesh_dict_path) 
    # TODO: 换数据集 bc5cdr_corpus_path
    sentences = get_corpus_sentences(nbci_corpus_path) 
    kp = init_keyword_processor(Mesh_dict) 
    matched_entities = match_entities(sentences, kp, Mesh_dict)  # Match entities in sentences
    #  TODO: 换数据集 bc5cdr_dictionary_output_dir_path
    save_matched_entities(matched_entities, nbci_dictionary_output_dir_path)  # 

    # 将text
    # result_dict = evaluate_ner(bc5cdr_dictionary_output_dir_path, bc5cdr_corpus_path) 
    # print(result_dict)
