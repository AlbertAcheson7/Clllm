import json
import string

from fragments_to_mention_exhaustive import fragments_to_mention_exhaustive
from fragments_to_mention_dependencyTree import dependency_tree_extraction
"""
cd /src/dictionary/preprocess 
python fragments_main.py
conda activate myenv
本脚本主要用于从/dictionary的输出数据转换到能输入到RAG的输入数据。
"""

def get_da_data(da_match_path):
    """
    :param da_match_path: str, da_match文件路径
    :return: list, processed data with fragments
    """
    with open(da_match_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        sentence = item.get('sentence')
        entities = item.get('entities', [])
        
        if not entities:
            item['fragments'] = [sentence] 
            continue

        entities_sorted = sorted(entities, key=lambda x: x[1])

        fragments = []
        prev_end = 0
        for entity in entities_sorted:
            start, end = entity[1], entity[2]
            if start > prev_end:
                fragments.append((sentence[prev_end:start]))
            # fragments.append((sentence[start:end]))
            prev_end = end

        tail = sentence[prev_end:]
        if prev_end < len(sentence) and any(c not in string.whitespace + string.punctuation for c in tail):
            fragments.append(tail)
        item['fragments'] = fragments
    
    return data

def save_result(data, output_path, method):
    result_data = []
    for item in data:
        fragments = item.get('fragments', [])
        
        if method == "exhaustive":
            mention_list_per_fragment = fragments_to_mention_exhaustive(fragments)
        elif method == "dependency_tree":
            mention_list_per_fragment = dependency_tree_extraction(fragments)
        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods: ['exhaustive', 'dependency_tree']")

        filtered_fragments = []
        filtered_mention_list = []
        for i, mentions in enumerate(mention_list_per_fragment):
            if mentions:
                filtered_fragments.append(fragments[i])
                filtered_mention_list.append(mentions)
        
        if not filtered_fragments:
            continue

        result_item = {
            'sentence_id': item.get('sentence_id', ''),
            'sentence': item.get('sentence', ''),
            'entities': item.get('entities', []),
            'fragments': filtered_fragments,
            'mention_list': filtered_mention_list
        }
        result_data.append(result_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    print(f"处理完成，共处理 {len(result_data)} 个sentence for method '{method}'")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # BC5CDR  将dictionary 的输出转换为fragments 文件 
    # ***************  dictionary 相关 ***************
    MESH_PATH = "/home/zhangzy/CLllm/data/data_pre/dict_pred/MeSH2017_desc2017_parsed_with_names.json"
    BC5CDR_DIR_PATH = "/home/zhangzy/CLllm/data/BC5CDR"
    NCBI_DIR_PATH = "/home/zhangzy/CLllm/data/NCBI"
    DICTIONARY_OUTPUT_DIR_PATH = "/home/zhangzy/CLllm/src/dictionary/output"

    # BC5CDR
    print("Processing BC5CDR dataset...")
    bc5cdr_da_match_path = DICTIONARY_OUTPUT_DIR_PATH + "/BC5CDR/bc5cdr_train_da.json"
    bc5cdr_data_with_fragments = get_da_data(bc5cdr_da_match_path) 
    save_result(bc5cdr_data_with_fragments, DICTIONARY_OUTPUT_DIR_PATH + "/bc5cdr_train_da_mention_dependencyTree_main_new.json", "dependency_tree")
    save_result(bc5cdr_data_with_fragments, DICTIONARY_OUTPUT_DIR_PATH + "/bc5cdr_train_da_mention_exhaustive_main_new.json", "exhaustive")
    print("-" * 20)

    # NCBI
    print("Processing NCBI dataset...")
    ncbi_da_match_path = DICTIONARY_OUTPUT_DIR_PATH + "/NCBI/ncbi_train_da.json"
    ncbi_data_with_fragments = get_da_data(ncbi_da_match_path) 
    save_result(ncbi_data_with_fragments, DICTIONARY_OUTPUT_DIR_PATH + "/ncbi_train_da_mention_dependencyTree_main_new.json", "dependency_tree")
    save_result(ncbi_data_with_fragments, DICTIONARY_OUTPUT_DIR_PATH + "/ncbi_train_da_mention_exhaustive_main_new.json", "exhaustive")
    print("-" * 20)

    


