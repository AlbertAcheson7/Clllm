import json
import re

# 读取JSON文件
with open('src/get_mention/ner_results__bc5cdr_formention_slide.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def find_entity_positions(text, entity_name):
    """在文本中找到实体的所有出现位置"""
    positions = []
    # 使用正则表达式进行不区分大小写的搜索
    for match in re.finditer(re.escape(entity_name), text, re.IGNORECASE):
        start_idx = match.start()
        end_idx = match.end()
        positions.append((start_idx, end_idx))
    return positions

# 处理每个条目
for entry in data:
    if 'gpt_ner_result' in entry and 'original_text' in entry:
        original_text = entry['original_text']
        old_gpt_ner = entry['gpt_ner_result']
        new_gpt_ner = []
        
        for item in old_gpt_ner:
            entity_name = item[0]
            entity_type = item[1]
            
            # 在原文中查找该实体的位置
            positions = find_entity_positions(original_text, entity_name)
            
            if positions:
                # 如果找到了位置，使用第一个匹配的位置
                start_idx, end_idx = positions[0]
                new_item = [entity_name, start_idx, end_idx, entity_type]
                new_gpt_ner.append(new_item)
            else:
                # 如果没找到，设置索引为-1（表示未找到）
                new_item = [entity_name, -1, -1, entity_type]
                new_gpt_ner.append(new_item)
                print(f"警告：在文本中未找到实体 '{entity_name}'")
        
        entry['gpt_ner_result'] = new_gpt_ner

# 保存更新后的文件
with open('src/get_mention/ner_results__bc5cdr_formention_slide.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("添加索引完成！")
print("新格式示例：")
if len(data) > 0:
    print("第一个条目的 gpt_ner_result:")
    for item in data[0]['gpt_ner_result']:
        print(f"  {item}")
    print("\n对比 slide_mention 格式：")
    for i, item in enumerate(data[0]['slide_mention'][:3]):
        print(f"  {item}") 