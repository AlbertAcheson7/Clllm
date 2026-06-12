import json
from collections import Counter

def verify_conversion(file_path: str):
    """验证转换后的数据格式并统计信息"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"=== 数据转换验证报告 ===")
    print(f"文档总数: {len(data)}")
    
    total_mentions = 0
    original_count = 0
    predicted_count = 0
    format_errors = 0
    position_errors = 0
    
    entity_types = Counter()
    
    for i, item in enumerate(data):
        mention_entities = item.get('mention_entity', [])
        total_mentions += len(mention_entities)
        
        for mention in mention_entities:
            # 检查格式
            if len(mention) != 6:
                format_errors += 1
                continue
            
            mention_text, start, end, entity_type, mesh_id, label = mention
            
            # 统计标签
            if label == "O":
                original_count += 1
            elif label == "P":
                predicted_count += 1
            
            # 统计实体类型
            entity_types[entity_type] += 1
            
            # 验证位置
            try:
                actual_text = item['text'][start:end]
                if actual_text != mention_text:
                    position_errors += 1
                    if position_errors <= 5:  # 只显示前5个错误
                        print(f"位置错误 - 文档{i+1}: 期望'{mention_text}', 实际'{actual_text}', 位置{start}:{end}")
            except:
                position_errors += 1
    
    print(f"\n=== 统计信息 ===")
    print(f"总mention数量: {total_mentions}")
    print(f"原始实体(O): {original_count}")
    print(f"候选mentions(P): {predicted_count}")
    print(f"格式错误: {format_errors}")
    print(f"位置错误: {position_errors}")
    
    print(f"\n=== 实体类型分布 ===")
    for entity_type, count in entity_types.most_common():
        print(f"{entity_type}: {count}")
    
    # 验证样例
    print(f"\n=== 样例验证 ===")
    sample_item = data[0]
    print(f"样例文本: {sample_item['text']}")
    print(f"mention_entity结构:")
    for i, mention in enumerate(sample_item['mention_entity'][:3]):
        print(f"  {i+1}. {mention}")

def main():
    file_path = '/home/zhangzy/CLllm/src/get_mention/converted_mention_result.json'
    verify_conversion(file_path)

if __name__ == "__main__":
    main() 