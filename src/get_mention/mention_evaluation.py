#!/usr/bin/env python3
"""
NER结果评估脚本 - 优化版本

主要功能：
1. 计算NER预测结果的精确率、召回率和F1分数
2. 支持两种评估模式：完全匹配、文本匹配
3. 分析实体类型分布和覆盖率
4. 提供详细的评估报告

作者：优化版本
时间：2025年01月25日
"""

import json
import logging
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    precision: float
    recall: float
    f1: float
    total_original: int
    total_predicted: int
    matched: int


@dataclass
class EvaluationConfig:
    """评估配置数据类"""
    debug_samples: int = 3
    analysis_samples: int = 100


class NEREntityExtractor:
    """实体提取器类"""
    
    @staticmethod
    def extract_original_entities(original_entities: List[List[Any]], include_type: bool = True) -> Set[Tuple]:
        """从原始实体列表中提取实体集合"""
        if include_type:
            return {(item[0], item[3]) for item in original_entities}
        else:
            return {item[0] for item in original_entities}
    
    @staticmethod
    def extract_predicted_entities(prediction_dict: Dict[str, List[str]], include_type: bool = True) -> Set[Tuple]:
        """从预测字典中提取实体集合"""
        entities = set()
        
        for entity_type, mentions in prediction_dict.items():
            for mention in mentions:
                if include_type:
                    entities.add((mention, entity_type))
                else:
                    entities.add(mention)
        
        return entities


class MetricsCalculator:
    """指标计算器类"""
    
    @staticmethod
    def calculate_metrics(original_entities: Set, predicted_entities: Set) -> EvaluationResult:
        """计算精确率、召回率和F1分数"""
        intersection = original_entities & predicted_entities
        matched = len(intersection)
        total_predicted = len(predicted_entities)
        total_original = len(original_entities)
        
        # 计算精确率
        precision = matched / total_predicted if total_predicted > 0 else 0.0
        
        # 计算召回率
        recall = matched / total_original if total_original > 0 else 0.0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            total_original=total_original,
            total_predicted=total_predicted,
            matched=matched
        )


class NERAnalyzer:
    """NER分析器类"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据文件"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"成功加载数据文件: {file_path}, 样本数量: {len(data)}")
            return data
        
        except Exception as e:
            self.logger.error(f"加载数据文件失败: {e}")
            raise
    
    def analyze_entity_types(self, data: List[Dict]) -> Dict[str, Any]:
        """分析实体类型分布"""
        original_types = set()
        predicted_types = set()
        
        analysis_data = data[:self.config.analysis_samples]
        
        for item in analysis_data:
            # 收集原始实体类型
            for entity in item['original_entities']:
                original_types.add(entity[3])
            
            # 收集预测实体类型
            for entity_type in item['gpt_ner_result'].keys():
                predicted_types.add(entity_type)
        
        return {
            'original_types': sorted(original_types),
            'predicted_types': sorted(predicted_types),
            'type_overlap': sorted(original_types & predicted_types),
            'only_in_original': sorted(original_types - predicted_types),
            'only_in_predicted': sorted(predicted_types - original_types)
        }
    
    def evaluate_exact_match(self, data: List[Dict]) -> EvaluationResult:
        """完全匹配评估（文本+类型）"""
        total_metrics = EvaluationResult(0, 0, 0, 0, 0, 0)
        
        for i, item in enumerate(data):
            original_entities = NEREntityExtractor.extract_original_entities(
                item['original_entities'], include_type=True
            )
            predicted_entities = NEREntityExtractor.extract_predicted_entities(
                item['gpt_ner_result'], include_type=True
            )
            
            metrics = MetricsCalculator.calculate_metrics(original_entities, predicted_entities)
            total_metrics = self._accumulate_metrics(total_metrics, metrics)
            
            # 调试输出
            if i < self.config.debug_samples:
                self._print_debug_info(i + 1, "完全匹配", original_entities, predicted_entities, metrics)
        
        return self._average_metrics(total_metrics, len(data))
    
    def evaluate_text_only(self, data: List[Dict]) -> EvaluationResult:
        """仅文本匹配评估（忽略类型）"""
        total_metrics = EvaluationResult(0, 0, 0, 0, 0, 0)
        
        for i, item in enumerate(data):
            original_entities = NEREntityExtractor.extract_original_entities(
                item['original_entities'], include_type=False
            )
            predicted_entities = NEREntityExtractor.extract_predicted_entities(
                item['gpt_ner_result'], include_type=False
            )
            
            metrics = MetricsCalculator.calculate_metrics(original_entities, predicted_entities)
            total_metrics = self._accumulate_metrics(total_metrics, metrics)
            
            # 调试输出
            if i < self.config.debug_samples:
                self._print_debug_info(i + 1, "文本匹配", original_entities, predicted_entities, metrics)
        
        return self._average_metrics(total_metrics, len(data))
    
    def calculate_coverage(self, data: List[Dict]) -> Dict[str, float]:
        """计算覆盖率统计"""
        total_not_covered = 0
        total_original = 0
        
        for item in data:
            original_mentions = {entity[0] for entity in item['original_entities']}
            predicted_mentions = {mention for mentions in item['gpt_ner_result'].values() 
                                for mention in mentions}
            
            not_covered = original_mentions - predicted_mentions
            total_not_covered += len(not_covered)
            total_original += len(original_mentions)
        
        coverage_rate = 1 - (total_not_covered / total_original) if total_original > 0 else 0
        
        return {
            'coverage_rate': coverage_rate,
            'total_not_covered': total_not_covered,
            'total_original': total_original
        }
    
    def evaluate_interval_coverage(self, data: List[Dict], alpha: int = 5, target_coverage: float = 0.95) -> Dict[str, Any]:
        """
        评估预测实体在扩展区间内对原始实体的覆盖情况
        
        Args:
            data: NER结果数据
            alpha: span扩展参数，预测实体span将向前向后各扩展alpha个span
            target_coverage: 目标覆盖率（默认95%）
            
        Returns:
            包含覆盖分析结果的字典
        """
        total_original_entities = 0
        covered_original_entities = 0
        samples_with_sufficient_coverage = 0
        coverage_details = []
        
        self.logger.info(f"开始基于span的区间覆盖评估，alpha={alpha}, 目标覆盖率={target_coverage:.1%}")
        
        for i, item in enumerate(data):
            original_text = item['original_text']
            original_entities = item['original_entities']
            predicted_entities = item['gpt_ner_result']
            
            # 获取所有预测实体的span位置
            predicted_spans = self._find_predicted_entity_spans(original_text, predicted_entities)
            
            # 生成基于span的扩展区间
            extended_intervals = self._generate_span_based_intervals(predicted_spans, alpha)
            
            # 检查原始实体覆盖情况
            sample_total = len(original_entities)
            sample_covered = 0
            uncovered_entities = []
            
            for orig_entity in original_entities:
                entity_text, start_pos, end_pos, entity_type, entity_id = orig_entity
                
                if self._is_entity_covered_by_intervals(start_pos, end_pos, extended_intervals):
                    sample_covered += 1
                else:
                    uncovered_entities.append({
                        'text': entity_text,
                        'start': start_pos,
                        'end': end_pos,
                        'type': entity_type
                    })
            
            sample_coverage = sample_covered / sample_total if sample_total > 0 else 1.0
            
            total_original_entities += sample_total
            covered_original_entities += sample_covered
            
            if sample_coverage >= target_coverage:
                samples_with_sufficient_coverage += 1
            
            # 调试输出
            if i < self.config.debug_samples:
                self.logger.info(f"样本 {i+1} 覆盖情况:")
                self.logger.info(f"  原始实体数: {sample_total}")
                self.logger.info(f"  覆盖实体数: {sample_covered}")
                self.logger.info(f"  覆盖率: {sample_coverage:.2%}")
                self.logger.info(f"  预测span数: {len(predicted_spans)}")
                self.logger.info(f"  扩展区间数: {len(extended_intervals)}")
                if uncovered_entities:
                    self.logger.info(f"  未覆盖实体: {uncovered_entities}")
            
            coverage_details.append({
                'sentence_id': item.get('sentence_id', f'sample_{i}'),
                'total_entities': sample_total,
                'covered_entities': sample_covered,
                'coverage_rate': sample_coverage,
                'uncovered_entities': uncovered_entities,
                'predicted_spans': predicted_spans,
                'extended_intervals': extended_intervals
            })
        
        overall_coverage = covered_original_entities / total_original_entities if total_original_entities > 0 else 0.0
        sufficient_coverage_rate = samples_with_sufficient_coverage / len(data) if data else 0.0
        
        return {
            'overall_coverage': overall_coverage,
            'total_original_entities': total_original_entities,
            'covered_original_entities': covered_original_entities,
            'samples_with_sufficient_coverage': samples_with_sufficient_coverage,
            'total_samples': len(data),
            'sufficient_coverage_rate': sufficient_coverage_rate,
            'target_coverage': target_coverage,
            'alpha': alpha,
            'meets_requirement': overall_coverage >= target_coverage,
            'coverage_details': coverage_details[:10]  # 只保留前10个样本的详细信息
        }
    
    def _find_predicted_entity_spans(self, text: str, predicted_entities: Dict[str, List[str]]) -> List[Tuple[int, int, str, str]]:
        """
        在文本中找到预测实体的所有span位置
        
        Returns:
            List of (start_pos, end_pos, entity_text, entity_type) sorted by start position
        """
        spans = []
        
        for entity_type, entity_list in predicted_entities.items():
            for entity_text in entity_list:
                # 使用正则表达式查找所有匹配位置（忽略大小写，单词边界匹配）
                pattern = r'\b' + re.escape(entity_text) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    spans.append((start_pos, end_pos, entity_text, entity_type))
        
        # 按位置排序
        spans.sort(key=lambda x: x[0])
        return spans
    
    def _generate_span_based_intervals(self, predicted_spans: List[Tuple[int, int, str, str]], alpha: int) -> List[Tuple[int, int]]:
        """
        基于span生成扩展区间
        对于每个预测span，向前向后各扩展alpha个span
        """
        if not predicted_spans:
            return []
        
        intervals = []
        
        for i, (start_pos, end_pos, entity_text, entity_type) in enumerate(predicted_spans):
            # 向前扩展alpha个span
            start_idx = max(0, i - alpha)
            # 向后扩展alpha个span  
            end_idx = min(len(predicted_spans) - 1, i + alpha)
            
            # 获取扩展范围内的起始和结束位置
            extended_start = predicted_spans[start_idx][0]  # 第一个span的开始位置
            extended_end = predicted_spans[end_idx][1]      # 最后一个span的结束位置
            
            intervals.append((extended_start, extended_end))
        
        # 合并重叠的区间
        return self._merge_intervals(intervals)
    
    def _find_predicted_entity_positions(self, text: str, predicted_entities: Dict[str, List[str]]) -> List[Tuple[int, int, str, str]]:
        """
        在文本中找到预测实体的所有位置（保留旧方法以兼容性）
        
        Returns:
            List of (start_pos, end_pos, entity_text, entity_type)
        """
        return self._find_predicted_entity_spans(text, predicted_entities)
    
    def _generate_extended_intervals(self, predicted_positions: List[Tuple[int, int, str, str]], 
                                   alpha: int, text_length: int) -> List[Tuple[int, int]]:
        """
        生成扩展区间（基于span的新实现）
        """
        return self._generate_span_based_intervals(predicted_positions, alpha)
    
    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的区间"""
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            
            if current_start <= last_end:
                # 重叠，合并区间
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # 不重叠，添加新区间
                merged.append((current_start, current_end))
        
        return merged
    
    def _is_entity_covered_by_intervals(self, entity_start: int, entity_end: int, 
                                      intervals: List[Tuple[int, int]]) -> bool:
        """检查实体是否被任何一个区间覆盖"""
        for interval_start, interval_end in intervals:
            if interval_start <= entity_start and entity_end <= interval_end:
                return True
        return False
    
    def _accumulate_metrics(self, total: EvaluationResult, current: EvaluationResult) -> EvaluationResult:
        """累积指标"""
        return EvaluationResult(
            precision=total.precision + current.precision,
            recall=total.recall + current.recall,
            f1=total.f1 + current.f1,
            total_original=total.total_original + current.total_original,
            total_predicted=total.total_predicted + current.total_predicted,
            matched=total.matched + current.matched
        )
    
    def _average_metrics(self, total: EvaluationResult, num_samples: int) -> EvaluationResult:
        """计算平均指标"""
        return EvaluationResult(
            precision=total.precision / num_samples,
            recall=total.recall / num_samples,
            f1=total.f1 / num_samples,
            total_original=total.total_original,
            total_predicted=total.total_predicted,
            matched=total.matched
        )
    
    def _print_debug_info(self, sample_num: int, mode: str, original: Set, predicted: Set, metrics: EvaluationResult):
        """打印调试信息"""
        print(f"\n--- 样本 {sample_num} ({mode}) ---")
        print(f"原始实体: {original}")
        print(f"预测实体: {predicted}")
        print(f"匹配实体: {original & predicted}")
        print(f"指标: P={metrics.precision:.4f}, R={metrics.recall:.4f}, F1={metrics.f1:.4f}")


class ReportGenerator:
    """报告生成器类"""
    
    @staticmethod
    def print_evaluation_report(file_path: str, data: List[Dict], analyzer: NERAnalyzer):
        """生成完整的评估报告"""
        print(f"\n{'='*60}")
        print(f"NER评估报告 - {Path(file_path).name}")
        print(f"数据样本数量: {len(data)}")
        print(f"{'='*60}")
        
        # 实体类型分析
        type_analysis = analyzer.analyze_entity_types(data)
        print(f"\n📊 实体类型分析:")
        print(f"  原始标注类型: {type_analysis['original_types']}")
        print(f"  预测结果类型: {type_analysis['predicted_types']}")
        print(f"  类型重叠: {type_analysis['type_overlap']}")
        print(f"  仅在原始中: {type_analysis['only_in_original']}")
        print(f"  仅在预测中: {type_analysis['only_in_predicted']}")
        
        # 评估模式1：完全匹配
        print(f"\n🎯 评估模式1: 完全匹配（文本+类型）")
        exact_result = analyzer.evaluate_exact_match(data)
        ReportGenerator._print_metrics("完全匹配", exact_result)
        
        # 评估模式2：仅文本匹配
        print(f"\n📝 评估模式2: 仅文本匹配（忽略类型）")
        text_result = analyzer.evaluate_text_only(data)
        ReportGenerator._print_metrics("文本匹配", text_result)
        
        # 覆盖率分析
        print(f"\n📈 覆盖率分析:")
        coverage = analyzer.calculate_coverage(data)
        print(f"  覆盖率: {coverage['coverage_rate']:.4f}")
        print(f"  未覆盖实体数: {coverage['total_not_covered']}")
        print(f"  总实体数: {coverage['total_original']}")
        
        # 区间覆盖分析
        print(f"\n🎯 区间覆盖分析:")
        ReportGenerator.print_interval_coverage_report(data, analyzer)
    
    @staticmethod
    def print_interval_coverage_report(data: List[Dict], analyzer: NERAnalyzer, alpha_values: Optional[List[int]] = None):
        """生成区间覆盖评估报告"""
        if alpha_values is None:
            alpha_values = [3, 5, 10, 15, 20]
        
        print(f"  测试不同的扩展参数 α 值:")
        print(f"  {'α值':<6} {'总体覆盖率':<12} {'≥95%样本率':<12} {'是否达标':<10}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
        
        best_alpha = None
        best_coverage = 0.0
        
        for alpha in alpha_values:
            coverage_result = analyzer.evaluate_interval_coverage(data, alpha=alpha, target_coverage=0.95)
            
            overall_coverage = coverage_result['overall_coverage']
            sufficient_rate = coverage_result['sufficient_coverage_rate']
            meets_requirement = coverage_result['meets_requirement']
            
            status = "✅ 是" if meets_requirement else "❌ 否"
            print(f"  {alpha:<6} {overall_coverage:<12.2%} {sufficient_rate:<12.2%} {status:<10}")
            
            if overall_coverage > best_coverage:
                best_coverage = overall_coverage
                best_alpha = alpha
        
        if best_alpha is not None:
            print(f"\n  📈 最佳扩展参数: α = {best_alpha} (覆盖率: {best_coverage:.2%})")
            
            # 详细分析最佳参数
            detailed_result = analyzer.evaluate_interval_coverage(data, alpha=best_alpha, target_coverage=0.95)
            ReportGenerator._print_detailed_interval_coverage(detailed_result)
    
    @staticmethod
    def _print_detailed_interval_coverage(coverage_result: Dict[str, Any]):
        """打印详细的区间覆盖结果"""
        print(f"\n  🔍 详细覆盖分析 (α = {coverage_result['alpha']}):")
        print(f"    总体覆盖率: {coverage_result['overall_coverage']:.2%}")
        print(f"    覆盖实体数: {coverage_result['covered_original_entities']}")
        print(f"    总实体数: {coverage_result['total_original_entities']}")
        print(f"    达到95%覆盖率的样本数: {coverage_result['samples_with_sufficient_coverage']}")
        print(f"    总样本数: {coverage_result['total_samples']}")
        print(f"    达标样本比例: {coverage_result['sufficient_coverage_rate']:.2%}")
        print(f"    是否满足95%要求: {'✅ 是' if coverage_result['meets_requirement'] else '❌ 否'}")
        
        # 显示部分样本详情
        if coverage_result['coverage_details']:
            print(f"\n  📋 样本覆盖详情 (前3个样本):")
            for i, detail in enumerate(coverage_result['coverage_details'][:3]):
                print(f"    样本 {i+1} ({detail['sentence_id']}):")
                print(f"      覆盖率: {detail['coverage_rate']:.2%} ({detail['covered_entities']}/{detail['total_entities']})")
                print(f"      预测span数: {len(detail['predicted_spans'])}")
                print(f"      扩展区间数: {len(detail['extended_intervals'])}")
                if detail['uncovered_entities']:
                    uncovered_texts = [e['text'] for e in detail['uncovered_entities'][:3]]
                    print(f"      未覆盖实体: {uncovered_texts[:3]}{'...' if len(detail['uncovered_entities']) > 3 else ''}")
    
    @staticmethod
    def _print_metrics(mode_name: str, result: EvaluationResult):
        """打印指标结果"""
        print(f"  精确率 (Precision): {result.precision:.4f}")
        print(f"  召回率 (Recall): {result.recall:.4f}")
        print(f"  F1分数: {result.f1:.4f}")
        print(f"  匹配实体数: {result.matched}")
        print(f"  预测实体总数: {result.total_predicted}")
        print(f"  标准实体总数: {result.total_original}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NER结果评估工具')
    parser.add_argument('--file1', default="/home/zhangzy/CLllm/src/get_mention/ner_results__bc5cdr_formention_20250715_131634.json",
                       help='第一个NER结果文件路径')
    parser.add_argument('--file2', default="/home/zhangzy/CLllm/src/get_mention/ner_results_bc5cdr_forentity_20250715_125626.json",
                       help='第二个NER结果文件路径')
    parser.add_argument('--debug-samples', type=int, default=3, help='调试输出的样本数量')
    parser.add_argument('--analysis-samples', type=int, default=100, help='类型分析的样本数量')
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvaluationConfig(
        debug_samples=args.debug_samples,
        analysis_samples=args.analysis_samples
    )
    
    # 创建分析器
    analyzer = NERAnalyzer(config)
    
    try:
        # 处理第一个文件
        data1 = analyzer.load_data(args.file1)
        ReportGenerator.print_evaluation_report(args.file1, data1, analyzer)
        
        # 处理第二个文件
        data2 = analyzer.load_data(args.file2)
        ReportGenerator.print_evaluation_report(args.file2, data2, analyzer)
        
        print(f"\n✅ 评估完成！")
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


"""
===========================================================
NER评估报告 - ner_results__bc5cdr_formention_20250715_131634.json
数据样本数量: 997
============================================================

📊 实体类型分析:
  原始标注类型: ['Chemical', 'Disease']
  预测结果类型: ['Chemical', 'Disease', 'raw_result']
  类型重叠: ['Chemical', 'Disease']
  仅在原始中: []
  仅在预测中: ['raw_result']

🎯 评估模式1: 完全匹配（文本+类型）
  精确率 (Precision): 0.3515
  召回率 (Recall): 0.5739
  F1分数: 0.4200
  匹配实体数: 2350
  预测实体总数: 7675
  标准实体总数: 3787

📝 评估模式2: 仅文本匹配（忽略类型）
  精确率 (Precision): 0.4496
  召回率 (Recall): 0.6551
  F1分数: 0.5135
  匹配实体数: 2644
  预测实体总数: 7009
  标准实体总数: 3787

📈 覆盖率分析:
  覆盖率: 0.6982
  未覆盖实体数: 1143
  总实体数: 3787
 
===========================================================
NER评估报告 - ner_results_bc5cdr_forentity_20250715_125626.json
数据样本数量: 997
============================================================

📊 实体类型分析:
  原始标注类型: ['Chemical', 'Disease']
  预测结果类型: ['Chemical', 'Disease']
  类型重叠: ['Chemical', 'Disease']
  仅在原始中: []
  仅在预测中: []

🎯 评估模式1: 完全匹配（文本+类型）
指标: P=0.5000, R=0.3333, F1=0.4000
  精确率 (Precision): 0.4752
  召回率 (Recall): 0.5286
  F1分数: 0.4746
  匹配实体数: 2162
  预测实体总数: 5484
  标准实体总数: 3787

📝 评估模式2: 仅文本匹配（忽略类型）
  精确率 (Precision): 0.5578
  召回率 (Recall): 0.6045
  F1分数: 0.5520
  匹配实体数: 2408
  预测实体总数: 5271
  标准实体总数: 3787

📈 覆盖率分析:
  覆盖率: 0.6359
  未覆盖实体数: 1379
  总实体数: 3787
  
✅ 评估完成！
"""