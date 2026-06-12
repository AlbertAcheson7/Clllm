import os
import json
import argparse
from typing import Any, Optional, List
from retrieve import Retriever, HIERARCHICAL_COLLECTION, SYNONYM_COLLECTION, DEFINITION_COLLECTION

# 配置（可根据需要修改或通过命令行传递）
CHROMA_DB_PATH = "/home/zhangzy/CLllm/src/rag_emc/index/chroma_db"
EMBEDDING_MODEL_PATH = "maidalun1020/bce-embedding-base_v1"
RERANKER_MODEL_PATH = "maidalun1020/bce-reranker-base_v1"
TOP_N_COARSE = 20
GENERATION_THRESHOLD = 0.5
MENTIONS_TO_PROCESS_LIMIT = None  # None 表示全部处理
DATA_TYPE_TO_LOAD = "mention_entity"  # 新的数据类型
RETRIEVAL_PROMPTS = {
    "hierarchical": "What is the broader category or type of the term {mention}?",
    "synonym": "What are the common synonyms of {mention}?",
    "definition": "What is {mention}?"
}

def extract_all_mentions_from_group(mention_group: List[Any]) -> List[str]:
    mentions = []
    if isinstance(mention_group, list):
        for mention in mention_group:
            if isinstance(mention, list) and len(mention) >= 1:
                mentions.append(mention[0])
    return mentions

def _extract_first_string(data: Any) -> Optional[str]:
    if isinstance(data, list):
        if len(data) > 0:
            return _extract_first_string(data[0])
        else:
            return None
    elif isinstance(data, str):
        return data
    else:
        return None

def load_mentions(file_path: str, limit: Optional[int]):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_mentions = []
    for item in data:
        if 'mention_list' in item and item['mention_list']:
            for mention_group in item['mention_list']:
                mentions_in_group = extract_all_mentions_from_group(mention_group)
                all_mentions.extend(mentions_in_group)
    if limit is not None:
        mentions_to_process = all_mentions[:limit]
        print(f"Processing the first {len(mentions_to_process)} mentions.")
    else:
        mentions_to_process = all_mentions
        print(f"Processing all {len(mentions_to_process)} mentions.")
    print(f"Sample mentions: {mentions_to_process[:5]}")
    return mentions_to_process

def load_entities(file_path: str, limit: Optional[int]):
    print(f"正在从 {file_path} 加载数据...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_entities = []
    for item in data:
        if 'entities' in item and item['entities']:
            for entity_structure in item['entities']:
                text = _extract_first_string(entity_structure)
                if text:
                    all_entities.append(text)
    seen = set()
    unique_entities = []
    for entity in all_entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)
    print(f"共发现 {len(all_entities)} 个实体实例，对应 {len(unique_entities)} 个唯一实体。")
    if limit is not None:
        entities_to_process = unique_entities[:limit]
        print(f"正在处理 {len(unique_entities)} 个唯一实体中的前 {len(entities_to_process)} 个。")
    else:
        entities_to_process = unique_entities
        print(f"正在处理所有 {len(entities_to_process)} 个唯一实体。")
    if entities_to_process:
        print(f"实体示例: {entities_to_process[:5]}")
    return entities_to_process

def load_mention_entity_data(file_path: str, limit: Optional[int]):
    """加载新格式的mention_entity数据"""
    print(f"Loading mention_entity data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_mentions = []
    for item in data:
        if 'mention_entity' in item and item['mention_entity']:
            for mention_data in item['mention_entity']:
                if isinstance(mention_data, list) and len(mention_data) >= 1:
                    mention_text = mention_data[0]  # 第一个元素是实体名称
                    if isinstance(mention_text, str) and mention_text.strip():
                        all_mentions.append(mention_text.strip())
    
    # 去重
    seen = set()
    unique_mentions = []
    for mention in all_mentions:
        if mention not in seen:
            seen.add(mention)
            unique_mentions.append(mention)
    
    print(f"共发现 {len(all_mentions)} 个实体提及，对应 {len(unique_mentions)} 个唯一实体。")
    
    if limit is not None:
        mentions_to_process = unique_mentions[:limit]
        print(f"Processing the first {len(mentions_to_process)} mentions.")
    else:
        mentions_to_process = unique_mentions
        print(f"Processing all {len(mentions_to_process)} mentions.")
    
    if mentions_to_process:
        print(f"Sample mentions: {mentions_to_process[:5]}")
    return mentions_to_process

def load_existing_retrieval_results(file_path: str) -> tuple[list, set]:
    processed_items = set()
    existing_results = []
    if not os.path.exists(file_path):
        print(f"检索结果文件 {file_path} 不存在，将从头开始检索。")
        return existing_results, processed_items
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            if not content.strip():
                print(f"检索结果文件 {file_path} 为空，将重新开始。")
                return [], set()
            existing_results = json.loads(content)
            if not isinstance(existing_results, list):
                print(f"警告：检索结果文件 {file_path} 格式不正确，应为JSON列表。将重新开始。")
                return [], set()
            for item in existing_results:
                if isinstance(item, dict) and 'mention' in item:
                    processed_items.add(item['mention'])
        except json.JSONDecodeError:
            print(f"警告：解析检索结果文件 {file_path} 时出错。将从头开始，并覆盖该文件。")
            return [], set()
    if processed_items:
        print(f"从 {file_path} 加载了 {len(processed_items)} 个已检索的项目。")
    return existing_results, processed_items

def main():
    parser = argparse.ArgumentParser(description="运行RAG检索流程。")
    parser.add_argument("--data_file_path", type=str, required=True, help="输入数据文件的路径。")
    parser.add_argument("--retrieval_output_path", type=str, required=True, help="用于保存检索结果的JSON文件的完整路径。")
    parser.add_argument("--log_name", type=str, default="run", help="本次运行的日志名称或标识符（当前未使用，为未来扩展保留）。")
    args = parser.parse_args()
    
    # 初始化检索器
    retriever = Retriever(
        embedding_model_path=EMBEDDING_MODEL_PATH,
        reranker_model_path=RERANKER_MODEL_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        device="cuda:0"
    )
    
    # 加载候选项目
    if DATA_TYPE_TO_LOAD == "mention_entity":
        candidate_items = load_mention_entity_data(args.data_file_path, MENTIONS_TO_PROCESS_LIMIT)
    elif DATA_TYPE_TO_LOAD == "mentions":
        candidate_items = load_mentions(args.data_file_path, MENTIONS_TO_PROCESS_LIMIT)
    elif DATA_TYPE_TO_LOAD == "entities":
        candidate_items = load_entities(args.data_file_path, MENTIONS_TO_PROCESS_LIMIT)
    else:
        raise ValueError("无效的 DATA_TYPE_TO_LOAD 配置。请选择 'mention_entity', 'mentions' 或 'entities'。")
    
    if not candidate_items:
        print("没有要处理的数据项，程序退出。")
        return
    
    # 加载已有的检索结果
    output_path = args.retrieval_output_path
    all_results, processed_items = load_existing_retrieval_results(output_path)
    items_to_process = [item for item in candidate_items if item not in processed_items]
    
    print(f"总共有 {len(candidate_items)} 个候选项目。")
    print(f"已检索 {len(processed_items)} 个项目。")
    print(f"本次运行将检索 {len(items_to_process)} 个新项目。")
    
    if not items_to_process:
        print("所有候选项目均已检索完毕，程序退出。")
        return
    
    # 执行检索
    for item in items_to_process:
        print(f"\n正在检索: {item}")
        retrieval_results = retriever.retrieve(
            item,
            TOP_N_COARSE,
            prompt_templates=RETRIEVAL_PROMPTS
        )
        
        # 计算最佳分数
        max_score = -1.0
        best_source = None
        has_results = False
        for source, result in retrieval_results.items():
            if result and result['document'] is not None:
                has_results = True
                if result['score'] > max_score:
                    max_score = result['score']
                    best_source = source
        
        if not has_results:
            print(f"No document found for '{item}' across all collections.")
            continue
        
        print(f"--- Overall Best Score from '{best_source}' is {max_score:.4f} ---")
        
        # 保存检索结果
        retrieval_item = {
            'mention': item,
            'retrieval_results': retrieval_results,
            'max_score': max_score,
            'best_source': best_source,
            'has_results': has_results,
            'threshold_passed': max_score > GENERATION_THRESHOLD
        }
        all_results.append(retrieval_item)
        
        # 实时保存结果
        if len(all_results) % 10 == 0:  # 每10个结果保存一次
            print(f"正在保存中间结果到 {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    # 最终保存
    print(f"\n检索完成，正在将 {len(all_results)} 条结果保存到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    # 统计信息
    threshold_passed_count = sum(1 for result in all_results if result.get('threshold_passed', False))
    print(f"检索完成统计:")
    print(f"  总检索项目: {len(all_results)}")
    print(f"  通过阈值({GENERATION_THRESHOLD})的项目: {threshold_passed_count}")
    print(f"  需要生成验证的项目: {threshold_passed_count}")
    print("Done.")

if __name__ == "__main__":
    main() 