import spacy
import re
import json

# conda activate myenv
def extract_from_fragments(fragments_list, nlp):
    mentions_list = []
    
    common_words = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }

    for frag in fragments_list:
        doc = nlp(frag)
        mentions = []
        
        # 1. 基于名词短语（noun chunks）作为候选mention
        for chunk in doc.noun_chunks:
            # 过滤过短或无意义的chunk可以在这里做，比如长度至少2个token
            if len(chunk) > 1:
                original_text = chunk.text
                # Clean the text: strip whitespace and common punctuation from ends
                mention_text = original_text.strip(' .,;:"\'()[]{}<>-')
                
                # Further filter out mentions that are likely not entities
                if not mention_text or mention_text.lower() in common_words:
                    continue
                # Filter out mentions that are just numbers or start with numbers and not part of a larger name.
                if re.match(r'^\d+$', mention_text) or (mention_text[0].isdigit() and not any(c.isalpha() for c in mention_text)):
                    continue

                start_offset = len(original_text) - len(original_text.lstrip(' .,;:"\'()[]{}<>-'))
                start_pos = chunk.start_char + start_offset
                end_pos = start_pos + len(mention_text)
                
                mentions.append([mention_text, start_pos, end_pos])
        
        # 2. 也可以加入基于实体的mention
        for ent in doc.ents:
            # 要求1: token长度 >=1 <=8
            if len(ent) < 1 or len(ent) > 8:
                continue
                
            # 要求2: 保留生物医学领域常用的实体类型
            valid_entity_types = {
                'CHEMICAL', 'DISEASE', 'GENE', 'PROTEIN', 'ORGANISM', 
                'ANATOMY', 'PROCESS', 'PHENOMENA', 'FUNCTION', 'CELL',
                'MOLECULE', 'COMPOUND', 'DRUG', 'MEDICATION', 'SYMPTOM'
            }
            if ent.label_ not in valid_entity_types:
                continue
            
            original_text = ent.text
            # Clean the entity text
            mention_text = original_text.strip(' .,;:"\'()[]{}<>-')
            if not mention_text:
                continue
                
            # 要求3: 内容过滤 - 排除标点符号等
            if re.match(r'^[\d\s\.,;:!?()\[\]{}"\'-]+$', mention_text):
                continue
                
            # 要求4: 排除全小写的常见词汇
            if mention_text.lower() in common_words:
                continue
            
            start_offset = len(original_text) - len(original_text.lstrip(' .,;:"\'()[]{}<>-'))
            start_pos = ent.start_char + start_offset
            end_pos = start_pos + len(mention_text)

            mentions.append([mention_text, start_pos, end_pos])
        
        # 去重
        if mentions:
            unique_mentions = [list(t) for t in {tuple(m) for m in mentions}]
            mentions_list.append(unique_mentions)
        else:
            mentions_list.append([])
    # print(mentions_list)
    return mentions_list

def dependency_tree_extraction(fragments_list):
    # 使用spacy进行依存句法分析
    nlp = spacy.load("en_core_sci_sm")
    return extract_from_fragments(fragments_list, nlp)


# test samples
""
# if __name__ == "__main__":
#     fragments_path = '/home/zhangzy/CLllm/src/dictionary/output/BC5CDR/bc5cdr_train_da_fragments.json'
#     output_path = '/home/zhangzy/CLllm/src/dictionary/output/bc5cdr_train_da_mention_dependencyTree.json'
    
#     # 读取fragments文件
#     with open(fragments_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 处理每个sentence的fragments
#     result_data = []
#     for item in data:
#         sentence = item.get('sentence', '')
#         entities = item.get('entities', [])
#         fragments = item.get('fragments', [])
#         sentence_id = item.get('sentence_id', '') # Keep sentence_id
        
#         # 为每个sentence的fragments生成mention
#         mention_list_per_fragment = dependency_tree_extraction(fragments)
        
#         # Filter out fragments that did not yield any mentions
#         filtered_fragments = []
#         filtered_mention_list = []
#         for i, mentions in enumerate(mention_list_per_fragment):
#             if mentions:  # Only keep if mentions list is not empty
#                 filtered_fragments.append(fragments[i])
#                 filtered_mention_list.append(mentions)
        
#         # 构建结果字典
#         result_item = {
#             'sentence_id': sentence_id,
#             'sentence': sentence,
#             'entities': entities,
#             'fragments': filtered_fragments,
#             'mention_list': filtered_mention_list
#         }
#         result_data.append(result_item)
    
#     # 保存结果到json文件
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(result_data, f, ensure_ascii=False, indent=4)
    
#     print(f"处理完成，共处理 {len(result_data)} 个sentence")
#     print(f"结果已保存到: {output_path}")
