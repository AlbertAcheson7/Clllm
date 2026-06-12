import json
import re
from typing import List, Dict, Any, Tuple

def find_mention_positions(text: str, mention: str) -> List[Tuple[int, int]]:
    """
    在文本中查找mention的所有位置
    
    Args:
        text: 完整文本
        mention: 要查找的mention字符串
    
    Returns:
        List of (start_idx, end_idx) tuples
    """
    positions = []
    start = 0
    
    while True:
        # 在文本中查找mention
        pos = text.find(mention, start)
        if pos == -1:
            break
        
        positions.append((pos, pos + len(mention)))
        start = pos + 1
    
    return positions

def get_mention_position_near_entity(text: str, mention: str, entity_start: int, entity_end: int) -> Tuple[int, int]:
    """
    获取mention在文本中靠近原始entity的位置
    
    Args:
        text: 完整文本
        mention: mention字符串
        entity_start: 原始entity的开始位置
        entity_end: 原始entity的结束位置
    
    Returns:
        (start_idx, end_idx) tuple
    """
    positions = find_mention_positions(text, mention)
    
    if not positions:
        # 如果找不到完全匹配，尝试模糊匹配
        return (entity_start, entity_end)
    
    if len(positions) == 1:
        return positions[0]
    
    # 如果有多个位置，选择最接近原始entity的那个
    entity_center = (entity_start + entity_end) // 2
    best_pos = positions[0]
    min_distance = abs((best_pos[0] + best_pos[1]) // 2 - entity_center)
    
    for pos in positions[1:]:
        pos_center = (pos[0] + pos[1]) // 2
        distance = abs(pos_center - entity_center)
        if distance < min_distance:
            min_distance = distance
            best_pos = pos
    
    return best_pos

def convert_mention_format(input_file: str, output_file: str):
    """
    转换mention格式：将original_entities和processed_entities合并成mention_entity
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    print(f"开始转换 {len(data)} 个文档...")
    
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(data)}")
        
        text = item['text']
        sentence_id = item['sentence_id']
        source = item['source']
        original_entities = item['original_entities']
        processed_entities = item['processed_entities']
        
        mention_entity = []
        
        # 处理每个实体
        for j, processed_entity in enumerate(processed_entities):
            original_entity = processed_entity['original_entity']
            candidate_mentions = processed_entity['candidate_mentions']
            
            entity_name, start_idx, end_idx, entity_type, mesh_id = original_entity
            
            # 1. 添加原始entity（标记为"O"）
            original_mention = [entity_name, start_idx, end_idx, entity_type, mesh_id, "O"]
            mention_entity.append(original_mention)
            
            # 2. 添加候选mentions（标记为"P"）
            for mention in candidate_mentions:
                # 跳过与原始entity相同的mention
                if mention == entity_name:
                    continue
                
                # 查找mention在文本中的位置
                mention_start, mention_end = get_mention_position_near_entity(
                    text, mention, start_idx, end_idx
                )
                
                candidate_mention = [mention, mention_start, mention_end, "No", mesh_id, "P"]
                mention_entity.append(candidate_mention)
        
        # 构建新的数据项
        converted_item = {
            'text': text,
            'sentence_id': sentence_id,
            'source': source,
            'mention_entity': mention_entity
        }
        
        converted_data.append(converted_item)
    
    # 保存转换后的数据
    print(f"正在保存转换后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共转换了 {len(converted_data)} 个文档")
    
    # 显示前几个示例
    print("\n=== 转换后的前2个示例 ===")
    for i, item in enumerate(converted_data[:2]):
        print(f"\n文档 {i+1}:")
        print(f"文本: {item['text']}")
        print(f"句子ID: {item['sentence_id']}")
        print(f"来源: {item['source']}")
        print("mention_entity:")
        
        for j, mention in enumerate(item['mention_entity']):
            mention_text, start, end, entity_type, mesh_id, label = mention
            print(f"  {j+1}. [{mention_text}, {start}, {end}, {entity_type}, {mesh_id}, {label}]")
            # 验证位置是否正确
            actual_text = item['text'][start:end]
            if actual_text != mention_text:
                print(f"      警告: 位置不匹配! 实际文本: '{actual_text}'")

def main():
    """主函数"""
    input_file = '/home/zhangzy/CLllm/src/get_mention/candidate_mentions_result.json'
    output_file = '/home/zhangzy/CLllm/src/get_mention/converted_mention_result.json'
    
    convert_mention_format(input_file, output_file)

if __name__ == "__main__":
    main() 