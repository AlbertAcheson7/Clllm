#!/usr/bin/env python3
"""
多环境RAG流水线运行脚本
支持为检索和生成阶段使用不同的conda环境
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command_with_env(cmd, description, conda_env=None):
    """在指定conda环境中运行命令"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    if conda_env:
        print(f"环境: {conda_env}")
        # 构建conda激活命令
        full_cmd = f"conda run -n {conda_env} " + " ".join(cmd)
        print(f"命令: {full_cmd}")
    else:
        full_cmd = " ".join(cmd)
        print(f"命令: {full_cmd}")
    print('='*60)
    
    try:
        if conda_env:
            # 使用conda run执行
            result = subprocess.run(
                ["conda", "run", "-n", conda_env] + cmd, 
                check=True, 
                capture_output=False
            )
        else:
            # 直接执行
            result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失败: {e}")
        return False

def check_conda_env(env_name):
    """检查conda环境是否存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return env_name in result.stdout
    except Exception as e:
        print(f"检查conda环境时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行多环境RAG检索-生成流水线")
    parser.add_argument("--data_file_path", type=str, required=True, 
                        help="输入数据文件的路径")
    parser.add_argument("--retrieval_output_path", type=str, required=True, 
                        help="检索结果保存路径")
    parser.add_argument("--validation_output_path", type=str, required=True, 
                        help="最终验证结果保存路径")
    parser.add_argument("--log_name", type=str, default="pipeline_run", 
                        help="运行日志名称")
    parser.add_argument("--skip_retrieval", action="store_true", 
                        help="跳过检索阶段，直接运行生成阶段")
    parser.add_argument("--retrieval_env", type=str, default="llama2-chn",
                        help="检索阶段使用的conda环境名称")
    parser.add_argument("--generation_env", type=str, default="llama3",
                        help="生成阶段使用的conda环境名称")
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    r_main_path = script_dir / "r_main.py"
    g_main_path = script_dir / "g_main.py"
    
    # 检查脚本是否存在
    if not r_main_path.exists():
        print(f"❌ 检索脚本不存在: {r_main_path}")
        return False
    if not g_main_path.exists():
        print(f"❌ 生成脚本不存在: {g_main_path}")
        return False
    
    # 检查conda环境
    if not args.skip_retrieval:
        if not check_conda_env(args.retrieval_env):
            print(f"❌ Conda环境不存在: {args.retrieval_env}")
            print("可用环境:")
            subprocess.run(["conda", "env", "list"])
            return False
    
    if not check_conda_env(args.generation_env):
        print(f"❌ Conda环境不存在: {args.generation_env}")
        print("可用环境:")
        subprocess.run(["conda", "env", "list"])
        return False
    
    success = True
    
    # 阶段1: 检索
    if not args.skip_retrieval:
        r_cmd = [
            "python", str(r_main_path),
            "--data_file_path", args.data_file_path,
            "--retrieval_output_path", args.retrieval_output_path,
            "--log_name", f"{args.log_name}_retrieval"
        ]
        
        success = run_command_with_env(
            r_cmd, 
            f"检索阶段(R) - 环境: {args.retrieval_env}", 
            args.retrieval_env
        )
        
        if not success:
            print("❌ 检索阶段失败，停止流水线")
            return False
            
        # 检查检索结果文件是否生成
        if not os.path.exists(args.retrieval_output_path):
            print(f"❌ 检索结果文件未生成: {args.retrieval_output_path}")
            return False
    else:
        print("⏭️  跳过检索阶段")
        # 检查检索结果文件是否存在
        if not os.path.exists(args.retrieval_output_path):
            print(f"❌ 检索结果文件不存在: {args.retrieval_output_path}")
            print("请先运行检索阶段或提供正确的检索结果文件路径")
            return False
    
    # 阶段2: 生成验证
    g_cmd = [
        "python", str(g_main_path),
        "--retrieval_input_path", args.retrieval_output_path,
        "--validation_output_path", args.validation_output_path,
        "--log_name", f"{args.log_name}_generation"
    ]
    
    success = run_command_with_env(
        g_cmd, 
        f"生成验证阶段(G) - 环境: {args.generation_env}", 
        args.generation_env
    )
    
    if success:
        print(f"\n🎉 完整流水线执行成功!")
        print(f"📁 检索结果文件: {args.retrieval_output_path}")
        print(f"📁 最终结果文件: {args.validation_output_path}")
        print(f"\n📊 环境使用情况:")
        print(f"  检索阶段: {args.retrieval_env}")
        print(f"  生成阶段: {args.generation_env}")
    else:
        print(f"\n❌ 生成验证阶段失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 