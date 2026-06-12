import os
import json
import argparse
import re
from collections import Counter
from typing import Any, Optional, List
from generation import Generator

# 配置（可根据需要修改或通过命令行传递）
GENERATION_MODEL_PATH = "/home/zhangzy/CLllm/model/meta-llama/Llama-3.1-8B-Instruct"
GENERATION_THRESHOLD = 0.5

def resolve_conflicts(validation_judgments: dict) -> str:
    votes = []
    entity_patterns = {
        "Chemical": r"\[?Chemical\]?",
        "Disease": r"\[?Disease\]?",
        "No": r"\[?No\]?"
    }
    for source, judgment in validation_judgments.items():
        answer = judgment.get('answer', '')
        found_entity = None
        for entity, pattern in entity_patterns.items():
            if re.search(pattern, answer, re.IGNORECASE):
                found_entity = entity
                break
        if found_entity:
            votes.append(found_entity)
    if not votes:
        return "Undetermined"
    vote_counts = Counter(votes)
    max_votes = max(vote_counts.values()) if vote_counts else 0
    top_candidates = [vote for vote, count in vote_counts.items() if count == max_votes]
    if len(top_candidates) == 1:
        return top_candidates[0]
    else:
        priority = {"Chemical": 3, "Disease": 2, "No": 1}
        top_candidates.sort(key=lambda x: priority.get(x, 99))
        return top_candidates[0]

def load_retrieval_results(file_path: str) -> List[dict]:
    """加载检索结果文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"检索结果文件 {file_path} 不存在。请先运行 r_main.py 生成检索结果。")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            if not content.strip():
                raise ValueError(f"检索结果文件 {file_path} 为空。")
            retrieval_results = json.loads(content)
            if not isinstance(retrieval_results, list):
                raise ValueError(f"检索结果文件 {file_path} 格式不正确，应为JSON列表。")
        except json.JSONDecodeError:
            raise ValueError(f"解析检索结果文件 {file_path} 时出错。")
    
    print(f"从 {file_path} 加载了 {len(retrieval_results)} 条检索结果。")
    return retrieval_results

def load_existing_validation_results(file_path: str) -> tuple[list, set]:
    """加载已有的验证结果"""
    processed_items = set()
    existing_results = []
    if not os.path.exists(file_path):
        print(f"验证结果文件 {file_path} 不存在，将从头开始验证。")
        return existing_results, processed_items
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            if not content.strip():
                print(f"验证结果文件 {file_path} 为空，将重新开始。")
                return [], set()
            existing_results = json.loads(content)
            if not isinstance(existing_results, list):
                print(f"警告：验证结果文件 {file_path} 格式不正确，应为JSON列表。将重新开始。")
                return [], set()
            for item in existing_results:
                if isinstance(item, dict) and 'mention' in item:
                    processed_items.add(item['mention'])
        except json.JSONDecodeError:
            print(f"警告：解析验证结果文件 {file_path} 时出错。将从头开始，并覆盖该文件。")
            return [], set()
    
    if processed_items:
        print(f"从 {file_path} 加载了 {len(processed_items)} 个已验证的项目。")
    return existing_results, processed_items

def main():
    parser = argparse.ArgumentParser(description="运行RAG生成验证流程。")
    parser.add_argument("--retrieval_input_path", type=str, required=True, help="检索结果JSON文件的路径。")
    parser.add_argument("--validation_output_path", type=str, required=True, help="用于保存验证结果的JSON文件的完整路径。")
    parser.add_argument("--log_name", type=str, default="run", help="本次运行的日志名称或标识符（当前未使用，为未来扩展保留）。")
    args = parser.parse_args()
    
    # 加载检索结果
    try:
        retrieval_results = load_retrieval_results(args.retrieval_input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return
    
    # 初始化生成器
    generator = Generator(model_path=GENERATION_MODEL_PATH)
    
    # 加载已有的验证结果
    output_path = args.validation_output_path
    all_results, processed_items = load_existing_validation_results(output_path)
    
    # 筛选需要验证的项目（通过阈值且未验证的）
    items_to_validate = []
    for retrieval_item in retrieval_results:
        mention = retrieval_item.get('mention')
        threshold_passed = retrieval_item.get('threshold_passed', False)
        
        if mention and threshold_passed and mention not in processed_items:
            items_to_validate.append(retrieval_item)
    
    # 统计信息
    total_retrieval_items = len(retrieval_results)
    threshold_passed_items = sum(1 for item in retrieval_results if item.get('threshold_passed', False))
    already_validated = len(processed_items)
    to_validate = len(items_to_validate)
    
    print(f"\n验证任务统计:")
    print(f"  总检索项目: {total_retrieval_items}")
    print(f"  通过阈值({GENERATION_THRESHOLD})的项目: {threshold_passed_items}")
    print(f"  已验证项目: {already_validated}")
    print(f"  本次需要验证的项目: {to_validate}")
    
    if not items_to_validate:
        print("没有需要验证的项目，程序退出。")
        return
    
    # 执行验证
    for i, retrieval_item in enumerate(items_to_validate, 1):
        mention = retrieval_item['mention']
        retrieval_results_data = retrieval_item['retrieval_results']
        max_score = retrieval_item['max_score']
        best_source = retrieval_item['best_source']
        
        print(f"\n[{i}/{to_validate}] 正在验证: {mention}")
        print(f"  最佳分数: {max_score:.4f} (来源: {best_source})")
        
        # 进行验证
        validation_result = generator.validate(
            mention, retrieval_results_data
        )
        
        # 解决冲突
        resolved_entity = resolve_conflicts(validation_result['validation_judgments'])
        validation_result['resolved_entity'] = resolved_entity
        
        print(f"  ==> 冲突解决实体: {resolved_entity}")
        
        # 添加检索信息到验证结果
        validation_result['max_score'] = max_score
        validation_result['best_source'] = best_source
        
        all_results.append(validation_result)
        
        # 实时保存结果
        if len(all_results) % 5 == 0:  # 每5个结果保存一次
            print(f"  正在保存中间结果到 {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    # 最终保存
    print(f"\n验证完成，正在将 {len(all_results)} 条结果保存到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    # 最终统计
    entity_counts = Counter(result.get('resolved_entity', 'Unknown') for result in all_results)
    print(f"\n验证完成统计:")
    print(f"  总验证项目: {len(all_results)}")
    print(f"  实体分布:")
    for entity, count in entity_counts.most_common():
        print(f"    {entity}: {count}")
    print("Done.")

if __name__ == "__main__":
    main() 
    # 构建新的generation 
    