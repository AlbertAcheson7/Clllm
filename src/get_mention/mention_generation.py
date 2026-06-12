# read /home/zhangzy/CLllm/src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json
# read 按照向左平移一个span 向右平移一个span 以及 同时向左向右平移一个span 生成新的mention
# 构造 "slide_mention" 字段 
# 保存到 src/get_mention/ner_results__bc5cdr_formention_slide.json

import json
from typing import List, Dict, Tuple, Set
import re
import string

def read_json_data(file_path: str) -> List[Dict]:
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def clean_word(word: str) -> str:
    """移除单词中的标点符号，用于匹配"""
    return word.strip(string.punctuation).lower()

def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """将文本分割成单词，返回(单词, 开始位置, 结束位置)的列表"""
    tokens = []
    # 使用正则表达式匹配单词（包括字母、数字、连字符等）
    pattern = r'\S+'  # 匹配非空白字符序列
    for match in re.finditer(pattern, text):
        word = match.group()
        start = match.start()
        end = match.end()
        tokens.append((word, start, end))
    return tokens

def find_mention_token_indices(tokens: List[Tuple[str, int, int]], mention: str) -> List[Tuple[int, int]]:
    """在token列表中找到mention对应的token索引范围"""
    mention_words = mention.split()
    if not mention_words:
        return []
    
    indices = []
    
    # 如果mention是单个词，尝试部分匹配
    if len(mention_words) == 1:
        mention_word = clean_word(mention_words[0])
        for i, (token_word, start_pos, end_pos) in enumerate(tokens):
            token_clean = clean_word(token_word)
            # 检查mention是否是token的一部分，或者token是否是mention的一部分
            if (mention_word in token_clean or 
                token_clean in mention_word or 
                token_clean == mention_word):
                indices.append((i, i))
    else:
        # 尝试精确的多词匹配（忽略标点符号）
        for i in range(len(tokens) - len(mention_words) + 1):
            # 检查从位置i开始的连续token是否匹配mention
            match = True
            for j, mention_word in enumerate(mention_words):
                if i + j >= len(tokens):
                    match = False
                    break
                token_word = tokens[i + j][0]
                # 忽略标点符号进行比较
                if clean_word(token_word) != clean_word(mention_word):
                    match = False
                    break
            
            if match:
                # 找到匹配，记录token索引范围
                start_idx = i
                end_idx = i + len(mention_words) - 1
                indices.append((start_idx, end_idx))
    
    return indices

def generate_slide_mentions(tokens: List[Tuple[str, int, int]], start_token_idx: int, end_token_idx: int, entity_type: str) -> List[List]:
    """生成滑动的mention：左移、右移、同时左右移动，返回[mention_text, start_char_idx, end_char_idx, entity_type]格式"""
    slide_mentions = []
    
    # 向左平移一个span（起始位置向左移动一个token）
    if start_token_idx > 0:
        left_slide_start = start_token_idx - 1
        left_slide_end = end_token_idx
        left_mention, start_char, end_char = extract_text_and_positions_from_tokens(tokens, left_slide_start, left_slide_end)
        if left_mention:
            slide_mentions.append([left_mention, start_char, end_char, entity_type])
    
    # 向右平移一个span（结束位置向右移动一个token）
    if end_token_idx < len(tokens) - 1:
        right_slide_start = start_token_idx
        right_slide_end = end_token_idx + 1
        right_mention, start_char, end_char = extract_text_and_positions_from_tokens(tokens, right_slide_start, right_slide_end)
        if right_mention:
            slide_mentions.append([right_mention, start_char, end_char, entity_type])
    
    # 同时向左向右平移一个span
    if start_token_idx > 0 and end_token_idx < len(tokens) - 1:
        both_slide_start = start_token_idx - 1
        both_slide_end = end_token_idx + 1
        both_mention, start_char, end_char = extract_text_and_positions_from_tokens(tokens, both_slide_start, both_slide_end)
        if both_mention:
            slide_mentions.append([both_mention, start_char, end_char, entity_type])
    
    return slide_mentions

def extract_text_and_positions_from_tokens(tokens: List[Tuple[str, int, int]], start_idx: int, end_idx: int) -> Tuple[str, int, int]:
    """从tokens中提取指定范围的文本和字符位置，保持原始格式"""
    if start_idx < 0 or end_idx >= len(tokens) or start_idx > end_idx:
        return "", -1, -1
    
    # 获取原始文本的起始和结束字符位置
    start_char_pos = tokens[start_idx][1]
    end_char_pos = tokens[end_idx][2]
    
    # 重构文本，保持token之间的原始间距
    text = ""
    for i in range(start_idx, end_idx + 1):
        if i > start_idx:
            # 添加token之间的空格（基于原始位置计算）
            prev_end = tokens[i-1][2]
            curr_start = tokens[i][1]
            if curr_start > prev_end:
                text += " " * (curr_start - prev_end)
        text += tokens[i][0]
    
    return text, start_char_pos, end_char_pos

def extract_text_from_tokens(tokens: List[Tuple[str, int, int]], start_idx: int, end_idx: int) -> str:
    """从tokens中提取指定范围的文本，保持原始格式（兼容性函数）"""
    text, _, _ = extract_text_and_positions_from_tokens(tokens, start_idx, end_idx)
    return text

def generate_mention_slides():
    """主函数：生成滑动的mention并保存结果"""
    
    # 1. 读取JSON文件
    input_file = "/home/zhangzy/CLllm/src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json"
    data = read_json_data(input_file)
    
    print(f"开始处理 {len(data)} 个句子...")
    
    # 处理每个句子
    for idx, item in enumerate(data):
        sentence_id = item["sentence_id"]
        original_text = item["original_text"]
        gpt_ner_result = item["gpt_ner_result"]
        
        # 将原文分割成tokens
        tokens = tokenize_text(original_text)
        
        # 为每个实体类型的mention生成slide_mention，保持gpt_ner_result原格式不变
        all_slide_mentions = []
        
        for entity_type, mentions in gpt_ner_result.items():
            for mention in mentions:
                # 找到mention在token列表中的位置
                token_indices = find_mention_token_indices(tokens, mention)
                
                # 对每个找到的位置生成滑动mention
                for start_token_idx, end_token_idx in token_indices:
                    slides = generate_slide_mentions(tokens, start_token_idx, end_token_idx, entity_type)
                    
                    # 将所有滑动结果添加到总列表中，避免重复
                    for slide_item in slides:
                        # 检查是否已存在相同文本和位置的mention（忽略entity_type）
                        slide_text, slide_start, slide_end, slide_type = slide_item
                        is_duplicate = False
                        for existing_item in all_slide_mentions:
                            existing_text, existing_start, existing_end, existing_type = existing_item
                            if slide_text == existing_text and slide_start == existing_start and slide_end == existing_end:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_slide_mentions.append(slide_item)
        
        # 添加slide_mention字段到原数据
        item["slide_mention"] = all_slide_mentions
        
        # 进度显示
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1} / {len(data)} 个句子")
    
    # 保存结果
    output_file = "src/get_mention/ner_results__bc5cdr_formention_slide.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 统计信息
    total_original_mentions = 0
    total_slide_mentions = 0
    
    for item in data[:10]:  # 分析前10个句子的统计
        gpt_ner_data = item.get("gpt_ner_result", {})
        for entity_type, mentions in gpt_ner_data.items():
            total_original_mentions += len(mentions)
        
        slide_data = item.get("slide_mention", [])
        total_slide_mentions += len(slide_data)
    
    print(f"\n=== 统计信息（前10个句子）===")
    print(f"原始mention数: {total_original_mentions}")
    print(f"生成的slide mention数: {total_slide_mentions}")
    
    # 显示一些示例
    print(f"\n=== 示例结果 ===")
    for i, item in enumerate(data[:3]):
        print(f"\n句子 {i+1}:")
        print(f"原文: {item['original_text'][:100]}...")
        print(f"原始NER结果:")
        gpt_ner_data = item.get("gpt_ner_result", {})
        for entity_type, mentions in gpt_ner_data.items():
            print(f"  {entity_type}: {mentions}")
        
        slide_data = item.get("slide_mention", [])
        print(f"滑动结果 (共{len(slide_data)}个):")
        for j, slide_item in enumerate(slide_data[:6]):  # 只显示前6个
            print(f"  {j+1}: {slide_item}")
        if len(slide_data) > 6:
            print(f"  ... 还有{len(slide_data)-6}个结果")

if __name__ == "__main__":
    generate_mention_slides()