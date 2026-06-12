import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
import matplotlib.pyplot as plt
from retriever import Retriever
from tqdm import tqdm

# --- 配置 (改编自 retriever.py) ---
CHROMA_DB_PATH = "/home/zhangzy/CLllm/src/rag_emc/index/chroma_db"
# 这是包含实体/提及的JSON数据文件的路径。
DATA_FILE_PATH = "/home/zhangzy/CLllm/src/dictionary/output/bc5cdr_train_da_mention_exhaustive.json"
COLLECTION_NAME = "mesh_knowledge"
EMBEDDING_MODEL_PATH = "maidalun1020/bce-embedding-base_v1"
RERANKER_MODEL_PATH = "maidalun1020/bce-reranker-base_v1"
TOP_N_COARSE = 20
# 设置为 None 以处理所有实体，或设置为一个整数以限制要处理的实体数量。
MENTIONS_TO_PROCESS_LIMIT = None
SCORES_OUTPUT_PATH = "retrieval_scores_entities.txt"
PLOT_OUTPUT_PATH = "retrieval_scores_distribution_entities.png"


def load_entities(file_path: str, limit: int | None):
    """
    从数据文件加载实体。
    假设JSON文件包含一个项目列表，每个项目都有一个'entities'列表，
    并且每个实体都有一个关联的实体名称。
    """
    print(f"正在从 {file_path} 加载数据...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_entities = []
    for item in data:
        if 'entities' in item and item['entities']:
            # 假设实体名称是每个子列表中的第4个元素（索引3）。
            all_entities.extend([entity[0] for entity in item['entities'] if len(entity) > 3])

    if not all_entities:
        print("未能找到符合预期格式的任何实体（'entities'列表，且条目长度 > 3）。")
        return []
    
    unique_entities = sorted(list(set(all_entities)))
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


def analyze_scores(scores: list[float]):
    """
    计算并打印分数的统计数据，生成并保存直方图。
    """
    if not scores:
        print("没有提供用于分析的分数。")
        return

    scores_array = np.array(scores)

    # --- 计算统计数据 ---
    mean_score = np.mean(scores_array)
    std_dev = np.std(scores_array)
    median_score = np.median(scores_array)
    percentiles = {p: np.percentile(scores_array, p) for p in [25, 75, 90, 95, 99]}
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    print("\n--- 相似度分数分布分析 ---")
    print(f"已分析的分数数量: {len(scores_array)}")
    print(f"平均值: {mean_score:.4f}")
    print(f"标准差: {std_dev:.4f}")
    print(f"最小值: {min_score:.4f}, 最大值: {max_score:.4f}")
    print(f"中位数 (第50个百分位数): {median_score:.4f}")
    for p, val in percentiles.items():
        print(f"第 {p} 个百分位数: {val:.4f}")

    # --- 绘制直方图 ---
    plt.figure(figsize=(12, 7))
    plt.hist(scores_array, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Top-1 重排检索分数分布', fontsize=16)
    plt.xlabel('重排器分数', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 添加平均值和中位数的垂直线
    plt.axvline(float(mean_score), color='r', linestyle='dashed', linewidth=2, label=f'平均值: {mean_score:.4f}')
    plt.axvline(float(median_score), color='g', linestyle='dashed', linewidth=2, label=f'中位数: {median_score:.4f}')
    plt.legend()

    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n直方图已保存至 {PLOT_OUTPUT_PATH}")
    plt.close()

    # --- 将原始分数保存到文件 ---
    with open(SCORES_OUTPUT_PATH, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")
    print(f"原始分数已保存至 {SCORES_OUTPUT_PATH}")


def main():
    """
    运行检索并分析分数的主函数。
    """
    # --- 1. 初始化检索器 ---
    retriever = Retriever(
        embedding_model_path=EMBEDDING_MODEL_PATH,
        reranker_model_path=RERANKER_MODEL_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        device="cuda:1"
    )

    # --- 2. 加载数据 ---
    entities_to_process = load_entities(DATA_FILE_PATH, MENTIONS_TO_PROCESS_LIMIT)
    if not entities_to_process:
        return

    # --- 3. 执行检索并收集分数 ---
    scores = []
    print("\n开始检索过程...")
    for entity in tqdm(entities_to_process, desc="为实体检索文档"):
        _, _, best_score = retriever.retrieve(entity, TOP_N_COARSE)
        # retrieve 方法会打印自己的日志，所以我们这里只处理分数。
        if best_score is not None and best_score != -1.0:
            scores.append(best_score)

    if not scores:
        print("没有成功的检索。无法执行分析。")
        return

    # --- 4. 分析和可视化分数 ---
    analyze_scores(scores)
    print("\n分析完成。")


if __name__ == "__main__":
    main() 