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
    """在token列表中找到mention对应的token索引范围，支持部分匹配，忽略标点符号"""
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

def generate_expansion_combinations(max_expand_size: int) -> List[Tuple[int, int]]:
    """生成所有可能的扩展组合 (左扩展, 右扩展)"""
    combinations = []
    
    # 生成所有可能的左右扩展组合
    for left_expand in range(max_expand_size + 1):
        for right_expand in range(max_expand_size + 1):
            # 至少要有一边扩展，且总扩展不超过max_expand_size
            if (left_expand > 0 or right_expand > 0) and (left_expand + right_expand <= max_expand_size):
                combinations.append((left_expand, right_expand))
    
    return combinations

def expand_mention_by_tokens(tokens: List[Tuple[str, int, int]], start_token_idx: int, end_token_idx: int, expand_size: int) -> List[str]:
    """基于token扩展mention，返回所有可能的扩展组合列表"""
    expanded_mentions = []
    
    # 生成所有可能的扩展组合
    combinations = generate_expansion_combinations(expand_size)
    
    for left_expand, right_expand in combinations:
        # 计算扩展后的token范围
        new_start_idx = max(0, start_token_idx - left_expand)
        new_end_idx = min(len(tokens) - 1, end_token_idx + right_expand)
        
        # 获取扩展后的文本
        if new_start_idx <= new_end_idx:
            expanded_text = ""
            for i in range(new_start_idx, new_end_idx + 1):
                if i > new_start_idx:
                    # 添加token之间的空格（基于原始位置计算）
                    prev_end = tokens[i-1][2]
                    curr_start = tokens[i][1]
                    if curr_start > prev_end:
                        expanded_text += " " * (curr_start - prev_end)
                expanded_text += tokens[i][0]
        else:
            expanded_text = ""
        
        if expanded_text and expanded_text not in expanded_mentions:
            expanded_mentions.append(expanded_text)
    
    return expanded_mentions

def check_coverage(expanded_mentions: List[str], original_entity: str) -> bool:
    """检查扩展后的mention列表是否有任何一个覆盖原始实体"""
    for expanded_mention in expanded_mentions:
        if original_entity.lower() in expanded_mention.lower():
            return True
    return False

def analyze_coverage():
    """主函数：分析扩展后的mention是否能够完全覆盖original_entities的实体"""
    
    # 1. 读取JSON文件
    file_path = "src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json"
    data = read_json_data(file_path)
    
    # 统计信息
    total_sentences = len(data)
    expand_sizes = [1, 2, 3,4,5]  # 左右扩展[1,2,3]个单词
    coverage_stats = {size: {"covered": 0, "total": 0} for size in expand_sizes}
    
    # 详细结果记录
    detailed_results = []
    
    print(f"开始分析 {total_sentences} 个句子...")
    
    for idx, item in enumerate(data):
        sentence_id = item["sentence_id"]
        original_text = item["original_text"]
        gpt_ner_result = item["gpt_ner_result"]
        original_entities = item["original_entities"]
        
        # 将原文分割成tokens
        tokens = tokenize_text(original_text)
        
        # 2. 获取所有的gpt_ner_result中的mention
        gpt_mentions = []
        for entity_type, mentions in gpt_ner_result.items():
            for mention in mentions:
                gpt_mentions.append(mention)
        
        # 获取original_entities中的实体名
        original_entity_names = [entity[0] for entity in original_entities]
        # 去重：按唯一实体计算
        unique_original_entities = list(set(original_entity_names))
        
        sentence_result = {
            "sentence_id": sentence_id,
            "original_text": original_text,
            "gpt_mentions": gpt_mentions,
            "original_entities": original_entity_names,  # 保留原始列表用于记录
            "unique_original_entities": unique_original_entities,  # 新增去重后的实体
            "coverage_results": {}
        }
        
        # 3. 对每个mention进行扩展并检查覆盖率
        for expand_size in expand_sizes:
            covered_entities = set()
            all_expanded_mentions = {}  # 记录所有扩展后的mention
            
            # 对每个GPT识别的mention进行处理
            for mention in gpt_mentions:
                # 找到mention在token列表中的位置
                token_indices = find_mention_token_indices(tokens, mention)
                
                mention_expansions = []
                for start_token_idx, end_token_idx in token_indices:
                    # 4. 基于token扩展mention，生成所有可能的组合
                    expanded_list = expand_mention_by_tokens(tokens, start_token_idx, end_token_idx, expand_size)
                    mention_expansions.extend(expanded_list)
                
                # 去重
                mention_expansions = list(set(mention_expansions))
                all_expanded_mentions[mention] = mention_expansions
                
                # 检查是否覆盖unique_original_entities（去重后的实体）
                for entity_name in unique_original_entities:
                    if check_coverage(mention_expansions, entity_name):
                        covered_entities.add(entity_name)
            
            # 计算覆盖率（基于去重后的实体）
            coverage_count = len(covered_entities)
            total_unique_entities = len(unique_original_entities)
            coverage_rate = coverage_count / total_unique_entities if total_unique_entities > 0 else 0
            
            sentence_result["coverage_results"][expand_size] = {
                "covered_entities": list(covered_entities),
                "coverage_count": coverage_count,
                "total_entities": len(original_entity_names),  # 原始总数（包含重复）
                "total_unique_entities": total_unique_entities,  # 去重后总数
                "coverage_rate": coverage_rate,  # 基于去重实体的覆盖率
                "expanded_mentions": all_expanded_mentions  # 添加所有扩展后的mention
            }
            
            # 更新统计（使用去重后的数据）
            coverage_stats[expand_size]["covered"] += coverage_count
            coverage_stats[expand_size]["total"] += total_unique_entities
        
        detailed_results.append(sentence_result)
        
        # 进度显示
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1} / {total_sentences} 个句子")
    
    # 输出统计结果
    print("\n=== 覆盖率统计结果 ===")
    for expand_size in expand_sizes:
        stats = coverage_stats[expand_size]
        overall_coverage = stats["covered"] / stats["total"] if stats["total"] > 0 else 0
        print(f"扩展大小 {expand_size} 个单词: {stats['covered']}/{stats['total']} = {overall_coverage:.4f} ({overall_coverage*100:.2f}%)")
    
    # 分析失败案例
    print("\n=== 失败案例分析（去重统计）===")
    uncovered_examples = []
    for result in detailed_results[:50]:  # 分析前50个句子
        expand_size = 3  # 使用扩展大小3来分析
        coverage_result = result["coverage_results"][expand_size]
        covered_entities = set(coverage_result["covered_entities"])
        # 使用去重后的实体
        all_unique_entities = set(result["unique_original_entities"])
        uncovered = all_unique_entities - covered_entities
        
        if uncovered:
            uncovered_examples.append({
                "sentence": result["original_text"],
                "gpt_mentions": result["gpt_mentions"],
                "original_entities": result["original_entities"],  # 保留原始（包含重复）
                "unique_entities": result["unique_original_entities"],  # 去重后的
                "uncovered": list(uncovered),
                "expanded_mentions": coverage_result["expanded_mentions"]
            })
    
    print(f"前50个句子中有 {len(uncovered_examples)} 个存在未覆盖实体（去重统计）")
    
    # GPT识别率分析
    print("\n=== GPT识别率分析 ===")
    total_original_entities = 0
    total_gpt_mentions = 0
    direct_matches = 0  # GPT识别的mention直接匹配原始实体的数量
    
    for result in detailed_results[:100]:  # 分析前100个句子
        # 使用去重后的实体进行分析
        unique_entities = result["unique_original_entities"]
        gpt_mentions = result["gpt_mentions"]
        
        total_original_entities += len(unique_entities)
        total_gpt_mentions += len(gpt_mentions)
        
        # 检查直接匹配（忽略大小写和标点符号）
        for entity in unique_entities:
            for mention in gpt_mentions:
                if clean_word(entity) == clean_word(mention) or clean_word(entity) in clean_word(mention) or clean_word(mention) in clean_word(entity):
                    direct_matches += 1
                    break  # 避免重复计算
    
    print(f"前100个句子统计（去重后）:")
    print(f"  原始唯一实体总数: {total_original_entities}")
    print(f"  GPT识别总数: {total_gpt_mentions}")
    print(f"  直接匹配数: {direct_matches}")
    print(f"  GPT识别覆盖率: {direct_matches/total_original_entities:.2%}")
    print(f"  GPT识别准确率: {direct_matches/total_gpt_mentions:.2%}" if total_gpt_mentions > 0 else "  GPT识别准确率: N/A")
    
    # 找出覆盖率最差的句子
    print("\n=== 覆盖率最差的10个句子（去重统计）===")
    sentence_coverage_stats = []
    
    for result in detailed_results:
        expand_size = 3  # 使用扩展大小3来分析
        coverage_result = result["coverage_results"][expand_size]
        # 使用去重后的实体数据
        total_unique_entities = coverage_result["total_unique_entities"]
        covered_entities = coverage_result["coverage_count"]
        coverage_rate = covered_entities / total_unique_entities if total_unique_entities > 0 else 1.0
        
        sentence_coverage_stats.append({
            "sentence_id": result["sentence_id"],
            "sentence": result["original_text"],
            "coverage_rate": coverage_rate,
            "covered": covered_entities,
            "total_unique": total_unique_entities,
            "total_original": coverage_result["total_entities"],  # 包含重复的原始总数
            "uncovered": total_unique_entities - covered_entities,
            "gpt_mentions": result["gpt_mentions"],
            "original_entities": result["original_entities"],
            "unique_entities": result["unique_original_entities"],
            "expanded_mentions": coverage_result["expanded_mentions"]
        })
    
    # 按覆盖率排序，取最差的10个
    worst_sentences = sorted(sentence_coverage_stats, key=lambda x: (x["coverage_rate"], -x["uncovered"]))[:10]
    
    for i, sentence_stat in enumerate(worst_sentences):
        print(f"\n最差句子 {i+1} (覆盖率: {sentence_stat['coverage_rate']:.2%}):")
        print(f"  ID: {sentence_stat['sentence_id']}")
        print(f"  原文: {sentence_stat['sentence'][:120]}...")
        print(f"  覆盖情况: {sentence_stat['covered']}/{sentence_stat['total_unique']} 个唯一实体 (原始: {sentence_stat['total_original']} 个)")
        print(f"  GPT识别: {len(sentence_stat['gpt_mentions'])} 个mention")
        print(f"  原始实体: {sentence_stat['original_entities']}")
        print(f"  去重实体: {sentence_stat['unique_entities']}")
        print(f"  GPT识别: {sentence_stat['gpt_mentions']}")
        
        # 分析未覆盖的实体（基于去重后的实体）
        covered_entities = set()
        for mention, expansions in sentence_stat['expanded_mentions'].items():
            for entity in sentence_stat['unique_entities']:
                if check_coverage(expansions, entity):
                    covered_entities.add(entity)
        
        uncovered_entities = set(sentence_stat['unique_entities']) - covered_entities
        if uncovered_entities:
            print(f"  未覆盖实体: {list(uncovered_entities)}")
            
        # 显示一些扩展结果
        print(f"  部分扩展结果:")
        for mention, expansions in list(sentence_stat['expanded_mentions'].items())[:3]:
            print(f"    '{mention}' -> {expansions[:2] if expansions else '[]'}")
    
    # 保存详细结果
    output_file = "src/get_mention/coverage_entity_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": coverage_stats,
            "detailed_results": detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return coverage_stats, detailed_results

if __name__ == "__main__":
    analyze_coverage()

"""
/home/zhangzy/CLllm/src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json
{
  "summary": {
    "1": {
      "covered": 3581,
      "total": 3787
    },
    "2": {
      "covered": 3656,
      "total": 3787
    },
    "3": {
      "covered": 3703,
      "total": 3787
    },
    "4": {
      "covered": 3729,
      "total": 3787
    },
    "5": {
      "covered": 3746,
      "total": 3787
    }
  }

"summary": {
    "1": {
      "covered": 3272,
      "total": 3787
    },
    "2": {
      "covered": 3405,
      "total": 3787
    },
    "3": {
      "covered": 3510,
      "total": 3787
    },
    "4": {
      "covered": 3575,
      "total": 3787
    },
    "5": {
      "covered": 3627,
      "total": 3787
    }

"""