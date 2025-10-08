#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果可视化模块

提供评估结果的可视化和报告生成功能。
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultVisualizer:
    """结果可视化工具"""
    
    @staticmethod
    def plot_metrics_comparison(
        evaluation_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ["map", "precision@10", "recall@10"],
        output_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """绘制多个策略的指标比较图
        
        Args:
            evaluation_results: 评估结果字典，策略名称到指标的映射
            metrics: 要比较的指标列表
            output_path: 输出文件路径，如果为None则显示图表
            figsize: 图表大小
        """
        try:
            # 准备数据
            data = []
            for strategy, metrics_dict in evaluation_results.items():
                for metric_name in metrics:
                    if metric_name in metrics_dict:
                        data.append({
                            "Strategy": strategy,
                            "Metric": metric_name,
                            "Value": metrics_dict[metric_name]
                        })
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning("没有数据可视化")
                return
            
            # 设置样式
            sns.set(style="whitegrid")
            
            # 创建图表
            plt.figure(figsize=figsize)
            chart = sns.barplot(x="Metric", y="Value", hue="Strategy", data=df)
            
            # 设置标题和标签
            plt.title("搜索策略性能比较", fontsize=16)
            plt.xlabel("评估指标", fontsize=12)
            plt.ylabel("分数", fontsize=12)
            
            # 添加数值标签
            for p in chart.patches:
                chart.annotate(f"{p.get_height():.4f}",
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom',
                              fontsize=8, rotation=0)
            
            # 调整布局
            plt.tight_layout()
            plt.legend(title="策略", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 保存或显示图表
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"图表已保存到: {output_path}")
            else:
                plt.show()
                
            plt.close()
        except Exception as e:
            logger.error(f"绘制比较图失败: {str(e)}")
    
    @staticmethod
    def generate_report(
        evaluation_results: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None
    ) -> str:
        """生成评估报告
        
        Args:
            evaluation_results: 评估结果字典，策略名称到指标的映射
            output_path: 输出文件路径，如果为None则只返回报告文本
            
        Returns:
            str: 报告文本
        """
        try:
            # 生成报告标题
            report = "# 搜索策略评估报告\n\n"
            
            # 添加概述
            report += "## 概述\n\n"
            report += f"本报告比较了 {len(evaluation_results)} 种搜索策略在 LitSearch 数据集上的性能。\n\n"
            
            # 找出最佳策略（基于MAP）
            best_strategy = max(evaluation_results.items(), key=lambda x: x[1].get("map", 0))[0]
            report += f"**最佳策略**: {best_strategy}\n\n"
            
            # 添加详细结果
            report += "## 详细结果\n\n"
            
            # 创建结果表格
            report += "| 策略 | MAP | Precision@1 | Precision@5 | Precision@10 | Recall@10 | 查询数量 |\n"
            report += "| ---- | --- | ----------- | ----------- | ------------ | --------- | -------- |\n"
            
            for strategy, metrics in evaluation_results.items():
                map_score = metrics.get("map", 0)
                p1 = metrics.get("precision@1", 0)
                p5 = metrics.get("precision@5", 0)
                p10 = metrics.get("precision@10", 0)
                r10 = metrics.get("recall@10", 0)
                query_count = metrics.get("query_count", 0)
                
                report += f"| {strategy} | {map_score:.4f} | {p1:.4f} | {p5:.4f} | {p10:.4f} | {r10:.4f} | {query_count} |\n"
            
            report += "\n"
            
            # 添加策略比较
            report += "## 策略比较\n\n"
            
            # 比较MAP
            report += "### MAP (平均精度均值)\n\n"
            strategies_by_map = sorted(evaluation_results.items(), key=lambda x: x[1].get("map", 0), reverse=True)
            for i, (strategy, metrics) in enumerate(strategies_by_map):
                report += f"{i+1}. **{strategy}**: {metrics.get('map', 0):.4f}\n"
            report += "\n"
            
            # 比较Precision@10
            report += "### Precision@10\n\n"
            strategies_by_p10 = sorted(evaluation_results.items(), key=lambda x: x[1].get("precision@10", 0), reverse=True)
            for i, (strategy, metrics) in enumerate(strategies_by_p10):
                report += f"{i+1}. **{strategy}**: {metrics.get('precision@10', 0):.4f}\n"
            report += "\n"
            
            # 比较Recall@10
            report += "### Recall@10\n\n"
            strategies_by_r10 = sorted(evaluation_results.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True)
            for i, (strategy, metrics) in enumerate(strategies_by_r10):
                report += f"{i+1}. **{strategy}**: {metrics.get('recall@10', 0):.4f}\n"
            report += "\n"
            
            # 添加结论
            report += "## 结论\n\n"
            report += f"基于评估结果，**{best_strategy}** 策略在整体性能上表现最佳，"
            report += f"其 MAP 分数为 {evaluation_results[best_strategy].get('map', 0):.4f}。\n\n"
            
            # 保存报告
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"报告已保存到: {output_path}")
            
            return report
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return "生成报告失败" 