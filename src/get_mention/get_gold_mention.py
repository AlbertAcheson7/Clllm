import json
import re
from typing import List, Tuple, Dict, Any

def get_candidate_mentions(text: str, entity_name: str, start_idx: int, end_idx: int) -> List[str]:
    """
    根据entity_name在文本中的位置，向左扩一个单词，向右扩一个单词，
    同时向左或向右扩一个单词，获取候选mention
    
    Args:
        text: 完整文本
        entity_name: 实体名称
        start_idx: 实体在文本中的开始位置
        end_idx: 实体在文本中的结束位置
    
    Returns:
        候选mention列表
    """
    candidates = []
    
    # 检查entity是否在括号中
    if start_idx > 0 and text[start_idx-1] == '(' and end_idx < len(text) and text[end_idx] == ')':
        return [entity_name]  # 如果在括号中，不扩展
    
    # 使用更精确的正则表达式来匹配单词（包括连字符）
    # 匹配由字母、数字、连字符组成的单词
    word_pattern = r'\b[\w-]+\b'
    word_positions = []
    
    # 找到每个单词在原文本中的位置
    for match in re.finditer(word_pattern, text):
        word_positions.append((match.start(), match.end(), match.group()))
    
    # 找到entity对应的单词索引范围
    entity_word_start = None
    entity_word_end = None
    
    for i, (word_start, word_end, word) in enumerate(word_positions):
        # 找到与entity重叠的单词
        if word_start < end_idx and word_end > start_idx:
            if entity_word_start is None:
                entity_word_start = i
            entity_word_end = i
    
    if entity_word_start is None or entity_word_end is None:
        return [entity_name]  # 如果找不到对应单词，返回原entity_name
    
    # 辅助函数：检查位置是否有标点符号阻止扩展
    def has_punctuation_before(pos: int) -> bool:
        if pos <= 0:
            return False
        # 检查前面是否有标点符号（逗号、句号、分号、感叹号、问号等）
        before_char = text[pos-1]
        return before_char in '.,;!?:)'
    
    def has_punctuation_after(pos: int) -> bool:
        if pos >= len(text):
            return False
        # 检查后面是否有标点符号
        after_char = text[pos]
        return after_char in '.,;!?:('
    
    # 生成候选mention
    # 1. 原始entity
    candidates.append(entity_name)
    
    # 2. 向左扩一个单词
    if entity_word_start > 0:
        # 检查entity前面是否有标点符号阻止扩展
        if not has_punctuation_before(word_positions[entity_word_start][0]):
            # 构建扩展的mention，保持原文本中的格式（包括连字符、空格等）
            left_word_start = word_positions[entity_word_start-1][0]
            entity_end = word_positions[entity_word_end][1]
            left_extended = text[left_word_start:entity_end]
            candidates.append(left_extended)
    
    # 3. 向右扩一个单词
    if entity_word_end < len(word_positions) - 1:
        # 检查entity后面是否有标点符号阻止扩展
        if not has_punctuation_after(word_positions[entity_word_end][1]):
            # 构建扩展的mention
            entity_start = word_positions[entity_word_start][0]
            right_word_end = word_positions[entity_word_end+1][1]
            right_extended = text[entity_start:right_word_end]
            candidates.append(right_extended)
    
    # 4. 同时向左右各扩一个单词
    if (entity_word_start > 0 and entity_word_end < len(word_positions) - 1 and
        not has_punctuation_before(word_positions[entity_word_start][0]) and
        not has_punctuation_after(word_positions[entity_word_end][1])):
        # 构建双向扩展的mention
        left_word_start = word_positions[entity_word_start-1][0]
        right_word_end = word_positions[entity_word_end+1][1]
        both_extended = text[left_word_start:right_word_end]
        candidates.append(both_extended)
    
    # 去重，保持顺序
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    
    return unique_candidates

def process_json_file(json_file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSON文件，遍历entities元素，生成候选mention
    
    Args:
        json_file_path: JSON文件路径
    
    Returns:
        处理后的数据，包含原始数据和候选mention
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        text = item['text']
        sentence_id = item['sentence_id']
        source = item['source']
        entities = item['entities']
        
        processed_entities = []
        
        for entity in entities:
            entity_name, start_idx, end_idx, entity_type, mesh_id = entity
            
            # 生成候选mention
            candidate_mentions = get_candidate_mentions(text, entity_name, start_idx, end_idx)
            
            processed_entity = {
                'original_entity': entity,
                'candidate_mentions': candidate_mentions
            }
            
            processed_entities.append(processed_entity)
        
        result_item = {
            'text': text,
            'sentence_id': sentence_id,
            'source': source,
            'original_entities': entities,
            'processed_entities': processed_entities
        }
        
        results.append(result_item)
    
    return results

def save_results_to_file(results: List[Dict[str, Any]], output_file_path: str):
    """
    将处理结果保存到JSON文件
    
    Args:
        results: 处理后的结果数据
        output_file_path: 输出文件路径
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file_path}")

def main():
    """主函数"""
    json_file_path = 'data/data_pre/data_pred/BC5CDR/merged_bc5cdr_test_data.json'
    
    print(f"正在处理文件: {json_file_path}")
    results = process_json_file(json_file_path)
    
    print(f"处理完成，共处理了 {len(results)} 个文档")
    
    # 保存结果到文件
    output_file_path = '/home/zhangzy/CLllm/src/get_mention/candidate_mentions_result.json'
    save_results_to_file(results, output_file_path)
    
    # 显示前几个示例
    print("\n=== 前3个示例 ===")
    for i, result in enumerate(results[:3]):
        print(f"\n文档 {i+1}:")
        print(f"文本: {result['text']}")
        print(f"句子ID: {result['sentence_id']}")
        print(f"来源: {result['source']}")
        print("实体和候选mention:")
        
        for j, entity in enumerate(result['processed_entities']):
            entity_name, start_idx, end_idx, entity_type, mesh_id = entity['original_entity']
            print(f"  实体 {j+1}: {entity_name} ({entity_type})")
            print(f"    位置: {start_idx}-{end_idx}")
            print(f"    候选mention: {entity['candidate_mentions']}")
    
    return results

if __name__ == "__main__":
    results = main()
