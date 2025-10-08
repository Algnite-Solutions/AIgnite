#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LitSearch 评估主程序

用于在命令行中运行评估。
"""

import os
import argparse
import logging
from typing import Dict, Any

from litsearch_evaluator import LitSearchEvaluator
from visualization import ResultVisualizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估LitSearch数据集上的搜索策略")
    
    # 基本参数
    parser.add_argument("--top_k", type=int, default=100, 
                       help="检索结果数量 (默认: 100)")
    parser.add_argument("--output_dir", type=str, default="./results", 
                       help="输出目录 (默认: ./results)")
    parser.add_argument("--sample_size", type=int, default=None, 
                       help="数据集采样大小，用于快速测试 (默认: 使用全部数据)")
    parser.add_argument("--quick_test", action="store_true",
                       help="快速测试模式，仅使用10篇文档和相关查询 (默认: False)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小 (默认: 32)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="日志级别 (默认: INFO)")
    
    # 数据库参数
    parser.add_argument("--init_metadata_db", type=bool, default=True,
                       help="是否初始化元数据数据库 (默认: True)")
    parser.add_argument("--db_host", type=str, default="localhost",
                       help="PostgreSQL 数据库主机 (默认: localhost)")
    parser.add_argument("--db_port", type=int, default=5432,
                       help="PostgreSQL 数据库端口 (默认: 5432)")
    parser.add_argument("--db_name", type=str, default="LitSearch",
                       help="PostgreSQL 数据库名称 (默认: LitSearch)")
    parser.add_argument("--db_user", type=str, default="postgres",
                       help="PostgreSQL 数据库用户名 (默认: postgres)")
    parser.add_argument("--db_password", type=str, default="11111",
                       help="PostgreSQL 数据库密码 (默认: 11111)")
    
    # 模型参数
    parser.add_argument("--init_vector_db", type=bool, default=True,
                       help="是否初始化向量数据库 (默认: True)")
    parser.add_argument("--vector_db_path", type=str, default="./litsearch_vector_db",
                       help="向量数据库路径 (默认: ./litsearch_vector_db)")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-base-en-v1.5",
                       help="向量嵌入模型名称 (默认: BAAI/bge-base-en-v1.5")
    parser.add_argument("--vector_weight", type=float, default=0.7,
                       help="混合搜索中向量搜索的权重 (默认: 0.7)")
    
    # GPU参数
    parser.add_argument("--gpu_devices", type=str, default='4,5',
                       help="指定要使用的 GPU 设备，例如 '6,7' (默认: 使用所有 GPU)")
    
    return parser.parse_args()

def main():
    """主程序"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 设置GPU设备
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        logger.info(f"设置 CUDA_VISIBLE_DEVICES={args.gpu_devices}")
    
    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"输出目录: {args.output_dir}")
    except Exception as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        return
    
    # 数据库配置
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "name": args.db_name,
        "user": args.db_user,
        "password": args.db_password
    }
    
    # 如果是快速测试模式，设置采样大小为10
    if args.quick_test:
        args.sample_size = 10
        logger.info("启用快速测试模式，采样大小设置为10")
    
    try:
        # 初始化评估器
        evaluator = LitSearchEvaluator(
            db_config=db_config,
            init_vector_db=args.init_vector_db,
            init_metadata_db=args.init_metadata_db,
            vector_db_path=args.vector_db_path, 
            embedding_model=args.embedding_model,
            vector_weight=args.vector_weight
        )
        
        # 运行评估
        evaluation_results = evaluator.run_evaluation(
            top_k=args.top_k,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
        
        # 生成比较图
        chart_path = os.path.join(args.output_dir, "strategies_comparison.png")
        logger.info(f"生成策略比较图: {chart_path}")
        ResultVisualizer.plot_metrics_comparison(
            evaluation_results,
            metrics=["map", "precision@1", "precision@5", "precision@10", "recall@10"],
            output_path=chart_path
        )
        
        # 生成报告
        report_path = os.path.join(args.output_dir, "evaluation_report.md")
        logger.info(f"生成评估报告: {report_path}")
        ResultVisualizer.generate_report(evaluation_results, output_path=report_path)
        
        logger.info(f"\n评估完成！结果已保存到 {args.output_dir} 目录")
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 