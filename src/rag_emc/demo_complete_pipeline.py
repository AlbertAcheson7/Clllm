#!/usr/bin/env python3
"""
完整RAG流水线演示脚本
展示从数据准备到最终结果的完整工作流程
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"🔹 {title}")
    print('='*60)

def print_results_summary(results_file):
    """打印结果摘要"""
    if not os.path.exists(results_file):
        print(f"❌ 结果文件不存在: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"📊 验证结果统计:")
    print(f"  总验证项目: {len(results)}")
    
    # 统计实体分布
    entity_counts = {}
    for result in results:
        entity = result.get('resolved_entity', 'Unknown')
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    print(f"  实体分布:")
    for entity, count in sorted(entity_counts.items()):
        print(f"    {entity}: {count}")
    
    # 显示一些示例
    print(f"\n📝 验证示例:")
    for i, result in enumerate(results[:3]):
        mention = result.get('mention', 'Unknown')
        resolved = result.get('resolved_entity', 'Unknown')
        max_score = result.get('max_score', 0)
        best_source = result.get('best_source', 'Unknown')
        print(f"  {i+1}. {mention} → {resolved} (分数: {max_score:.4f}, 来源: {best_source})")

def main():
    print_section("RAG 分离式流水线完整演示")
    
    # 检查环境
    print("🔍 检查系统环境...")
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    
    required_envs = ['llama2-chn', 'llama3']
    available_envs = result.stdout
    
    for env in required_envs:
        if env in available_envs:
            print(f"  ✅ {env} 环境可用")
        else:
            print(f"  ❌ {env} 环境不可用")
            print(f"请创建或检查conda环境: {env}")
            return False
    
    # 文件路径设置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_dir = Path("demo_results")
    demo_dir.mkdir(exist_ok=True)
    
    data_file = "test_data_3samples.json"
    retrieval_results = demo_dir / f"retrieval_results_{timestamp}.json"
    validation_results = demo_dir / f"validation_results_{timestamp}.json"
    
    print_section("步骤1: 数据准备")
    print(f"📁 测试数据: {data_file}")
    print(f"📁 检索结果: {retrieval_results}")
    print(f"📁 验证结果: {validation_results}")
    
    if not os.path.exists(data_file):
        print(f"❌ 测试数据文件不存在: {data_file}")
        print("请先运行之前的测试或创建测试数据文件")
        return False
    
    # 显示测试数据内容
    with open(data_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"📋 测试数据包含 {len(test_data)} 个文档:")
    total_mentions = 0
    for doc in test_data:
        mention_count = len(doc.get('mention_list', []))
        total_mentions += mention_count
        print(f"  - {doc.get('pmid', 'Unknown')}: {mention_count} 个实体")
    print(f"📝 总共 {total_mentions} 个待验证实体")
    
    print_section("步骤2: 运行完整流水线")
    
    # 构建命令
    cmd = [
        "python", "run_pipeline_multi_env.py",
        "--data_file_path", data_file,
        "--retrieval_output_path", str(retrieval_results),
        "--validation_output_path", str(validation_results),
        "--log_name", f"demo_{timestamp}"
    ]
    
    print(f"🚀 执行命令: {' '.join(cmd)}")
    print("⏳ 开始运行流水线...")
    
    try:
        # 运行流水线
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 流水线执行成功!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 流水线执行失败: {e}")
        return False
    
    print_section("步骤3: 结果分析")
    
    # 检查输出文件
    if retrieval_results.exists():
        print(f"✅ 检索结果文件生成: {retrieval_results}")
        with open(retrieval_results, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        print(f"📊 检索了 {len(retrieval_data)} 个实体")
        
        # 统计通过阈值的项目
        passed_threshold = sum(1 for item in retrieval_data if item.get('threshold_passed', False))
        print(f"📈 通过阈值的项目: {passed_threshold}/{len(retrieval_data)}")
    else:
        print(f"❌ 检索结果文件未生成")
    
    if validation_results.exists():
        print(f"✅ 验证结果文件生成: {validation_results}")
        print_results_summary(str(validation_results))
    else:
        print(f"❌ 验证结果文件未生成")
    
    print_section("演示完成")
    print("🎉 RAG分离式流水线演示成功完成!")
    print(f"📁 结果文件保存在: {demo_dir}")
    print("\n💡 后续步骤:")
    print("  1. 检查验证结果的准确性")
    print("  2. 调整阈值和参数优化性能")
    print("  3. 在大规模数据上运行完整流水线")
    print("  4. 分析错误案例并改进模型")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 