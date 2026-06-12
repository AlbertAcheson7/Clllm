import json
import re
from collections import defaultdict

def find_token_indices(text, tokens):
    indices = []
    start = 0
    for token in tokens:
        try:
            # Find the token starting from the last position
            index = text.index(token, start)
            indices.append((index, index + len(token)))
            start = index + len(token)
        except ValueError:
            # This case handles situations where tokenization might not perfectly match string slicing,
            # for example with complex unicode or punctuation rules.
            # A more robust solution might involve regex with word boundaries.
            # For this implementation, we assume simple whitespace tokenization is sufficient.
            pass  # Or handle error appropriately
    return indices

def fragments_to_mention_exhaustive(fragments_list):
    """
    :param fragments_list: list, a list of fragment strings
    :return: list of lists, where each inner list contains mentions for a fragment
    """
    all_fragments_mentions = [[] for _ in fragments_list]
    
    token_lists = []
    for frag in fragments_list:
        stripped_frag = frag.strip()
        tokens = stripped_frag.split()
        token_lists.append(tokens)

    global_idx = 0
    token_spans = []  # [(fragment_id, local_start, token, global_index)]
    for frag_id, tokens in enumerate(token_lists):
        for local_idx, token in enumerate(tokens):
            token_spans.append((frag_id, local_idx, token, global_idx))
            global_idx += 1

    valid_tokens = [t for t in token_spans if 1 <= t[3] < 8]

    group_by_frag = defaultdict(list)
    for t in valid_tokens:
        group_by_frag[t[0]].append(t)
    
    for frag_id, tokens_in_frag in group_by_frag.items():
        if not tokens_in_frag:
            continue
            
        original_frag = fragments_list[frag_id]
        
        # We need to map tokens back to their char positions in the original fragment string
        # This is non-trivial if tokens can be repeated.
        # A simple `split()` and then `find()` is not robust.
        # Let's try to get positions more reliably.
        
        # We assume tokens in `tokens_in_frag` are ordered by their appearance.
        text_tokens = [t[2] for t in tokens_in_frag]
        
        # A simplified way to find char spans, assuming tokens appear in order
        current_pos = 0
        token_char_spans = {} # (frag_id, local_idx) -> (start, end)
        
        # Generate char spans for all tokens in the fragment first
        frag_tokens = token_lists[frag_id]
        frag_text = fragments_list[frag_id].strip()
        
        last_pos = 0
        all_token_indices = []
        for token in frag_tokens:
            try:
                start = frag_text.index(token, last_pos)
                end = start + len(token)
                all_token_indices.append((start, end))
                last_pos = end
            except ValueError:
                all_token_indices.append((-1, -1)) # Should not happen if tokenization is simple split

        # Map local index to char span
        local_idx_to_span = {i: span for i, span in enumerate(all_token_indices)}

        n = len(tokens_in_frag)
        for i in range(n):
            for j in range(i, n):
                mention_tokens_info = tokens_in_frag[i:j+1]
                mention_text = " ".join(t[2] for t in mention_tokens_info)
                
                # Requirement 1: Filter out undesirable mentions
                if mention_text.strip() in {",", ".", ", .", ". ,"}:
                    continue
                if not mention_text.strip():
                    continue

                # Requirement 4: Get char positions
                first_token_local_idx = mention_tokens_info[0][1]
                last_token_local_idx = mention_tokens_info[-1][1]
                
                start_pos = local_idx_to_span.get(first_token_local_idx, (-1,-1))[0]
                end_pos = local_idx_to_span.get(last_token_local_idx, (-1,-1))[1]

                if start_pos != -1 and end_pos != -1:
                    all_fragments_mentions[frag_id].append([mention_text, start_pos, end_pos])
    
    return all_fragments_mentions

# if __name__ == "__main__":
#     fragments_path = '/home/zhangzy/CLllm/src/dictionary/output/BC5CDR/bc5cdr_train_da_fragments.json'
#     output_path = '/home/zhangzy/CLllm/src/dictionary/output/bc5cdr_train_da_mention_exhaustive_new.json'
    
#     with open(fragments_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     result_data = []
#     for item in data:
#         sentence = item.get('sentence', '')
#         entities = item.get('entities', [])
#         fragments = item.get('fragments', [])
#         sentence_id = item.get('sentence_id', '')

#         mention_list_per_fragment = fragments_to_mention_exhaustive(fragments)
        
#         filtered_fragments = []
#         filtered_mention_list = []
#         for i, mentions in enumerate(mention_list_per_fragment):
#             if mentions:
#                 filtered_fragments.append(fragments[i])
#                 filtered_mention_list.append(mentions)
        
#         if not filtered_fragments:
#             continue

#         result_item = {
#             'sentence_id': sentence_id,
#             'sentence': sentence,
#             'entities': entities,
#             'fragments': filtered_fragments,
#             'mention_list': filtered_mention_list
#         }
#         result_data.append(result_item)
    
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(result_data, f, ensure_ascii=False, indent=4)
    
#     print(f"处理完成，共处理 {len(result_data)} 个sentence")
#     print(f"结果已保存到: {output_path}")
    

