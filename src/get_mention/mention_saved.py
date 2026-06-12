import json
import re
from typing import List, Dict, Tuple, Set

def read_json_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sentence_to_words(sentence: str) -> List[str]:
    # 将句子分割为单词
    return re.findall(r'\w+|[^\w\s]', sentence)  # ['Famotidine', '-', 'associated', 'delirium', '.', 'A', 'series', 'of', 'six', 'cases', '.']

def gpt_result_to_list(gpt_ner_result: Dict[str, List[str]]) -> List[Tuple[List[str], str]]:
    """
    Converts GPT NER result to a list of tuples.

    Example input:
     gpt_ner_result = {
        "Disease": ["Famotidine", "delirium"],
        "Chemical": ["Famotidine"]
    }
    
    Example output:
    [ (['Famotidine'], 'Disease'), (['delirium'], 'Disease'), (['Famotidine'], 'Chemical') ]
    """
    all_mentions = []
    for entity_type, mentions in gpt_ner_result.items():
        if mentions:
            for mention in mentions:
                mention_words = sentence_to_words(mention)
                all_mentions.append((mention_words, entity_type))
    return all_mentions

def get_mention_words_indices(sentence_words: List[str], mention_words: List[str]) -> List[int]:
    
    if not mention_words:
        return []
    
    mention_words_indices = []

    # if len(mention_words) == 1:
    #     for (i, )
              

def main(path):
    data = read_json_data(path)  # 加载json文件
    
    for idx, item in enumerate(data):
        sentence_id = item["sentence_id"]
        original_text = item["original_text"]
        gpt_ner_result = item["gpt_ner_result"]

        sentence_words = sentence_to_words(original_text)  # 分割单词 ['Famotidine', '-', 'associated', 'delirium', '.', 'A', 'series', 'of', 'six', 'cases', '.']

        all_slide_mentions = []

        gpt_ner_result_words = gpt_result_to_list(gpt_ner_result) # [(['Famotidine'], 'Disease'), (['delirium'], 'Disease'), (['Famotidine'], 'Chemical')]

        for mention_words, entity_type in gpt_ner_result_words:
            # 找到mention_words在sentence_words中的位置
            mention_words_indices = get_mention_words_indices(sentence_words, mention_words)



    pass

if __name__ == "__main__":
    path_gpt = "/home/zhangzy/CLllm/src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json"

    main(path_gpt)



   









        
