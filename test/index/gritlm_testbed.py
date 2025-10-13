#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GritLM专用测试床

继承自TestBed，专门用于GritLM功能测试。
使用test_gritlm.py中的4篇真实文章进行测试。
"""

from recommendation.testbed.base_testbed import TestBed
from AIgnite.index.paper_indexer import PaperIndexer
from AIgnite.data.docset import DocSet, TextChunk, ChunkType
from AIgnite.db.metadata_db import MetadataDB, Base
from AIgnite.db.vector_db import VectorDB
from sqlalchemy import create_engine, text, inspect

from typing import Dict, Any, Tuple, List, Optional
import os
import sys
import unittest
from pathlib import Path
import logging

def setup_logging(level: str = "INFO") -> None:
    """设置日志配置
    
    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gritlm_testbed.log')
        ]
    )

logger = logging.getLogger(__name__)

class GritLMTestBed(TestBed):
    """GritLM专用测试床
    
    提供完整的GritLM功能测试，包括：
    - 文档索引测试
    - 向量搜索测试
    - 指令效果测试
    - 相似度排序测试
    """
    
    def __init__(self, config_path: str):
        """初始化GritLM测试床
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        self.indexer = None
        self.test_papers = []
    
    def check_environment(self) -> Tuple[bool, str]:
        """检查测试环境
        
        Returns:
            Tuple[bool, str]: (是否就绪, 错误信息)
        """
        try:
            # 检查配置文件
            if not os.path.exists(self.config_path):
                return False, f"Config file not found: {self.config_path}"
            
            # 检查数据库连接
            db_url = self.config['metadata_db']['db_url']
            if not db_url:
                return False, "Database URL not found in config"
            
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            
            # 检查向量数据库路径权限
            vector_db_path = self.config['vector_db']['db_path']
            if not vector_db_path:
                return False, "Vector database path not found in config"
            
            vector_db_dir = Path(vector_db_path).parent
            vector_db_dir.mkdir(parents=True, exist_ok=True)
            
            # 测试写权限
            test_file = vector_db_dir / "test_write_permission"
            test_file.write_text("test")
            test_file.unlink()
            
            return True, "Environment check passed"
            
        except Exception as e:
            return False, f"Environment check failed: {str(e)}"
    
    def load_data(self) -> List[DocSet]:
        """加载测试数据 - 基于test_gritlm.py中的4篇文章
        
        Returns:
            测试论文数据列表
        """
        self.logger.info("Creating test papers based on test_gritlm.py...")
        
        # 基于test_gritlm.py中的4篇文章创建DocSet对象
        self.test_papers = [
            DocSet(
                doc_id="13520958",
                title="A Simple but Powerful Automatic Term Extraction Method",
                abstract="In this paper, we propose a new idea for the automatic recognition of domain specific terms. Our idea is based on the statistics between a compound noun and its component single-nouns. More precisely, we focus basically on how many nouns adjoin the noun in question to form compound nouns. We propose several scoring methods based on this idea and experimentally evaluate them on the NTCIR1 TMREC test collection. The results are very promising especially in the low recall area.",
                authors=["Author 1", "Author 2"],
                categories=["cs.CL", "cs.IR"],
                published_date="2021-06-28",
                text_chunks=[
                    TextChunk(id="chunk1", type=ChunkType.TEXT, text="Our idea is based on the statistics between a compound noun and its component single-nouns."),
                    TextChunk(id="chunk2", type=ChunkType.TEXT, text="We propose several scoring methods based on this idea and experimentally evaluate them."),
                    TextChunk(id="chunk3", type=ChunkType.TEXT, text="The results are very promising especially in the low recall area.")
                ],
                figure_chunks=[],
                table_chunks=[],
                metadata={},
                pdf_path="dummy_path_1.pdf",
                HTML_path=None,
                comments='Test paper 1 - Term Extraction'
            ),
            DocSet(
                doc_id="8140780",
                title="Stronger Baselines for Trustable Results in Neural Machine Translation",
                abstract="Interest in neural machine translation has grown rapidly as its effectiveness has been demonstrated across language and data scenarios. New research regularly introduces architectural and algorithmic improvements that lead to significant gains over 'vanilla' NMT implementations. However, these new techniques are rarely evaluated in the context of previously published techniques, specifically those that are widely used in state-of-theart production and shared-task systems. As a result, it is often difficult to determine whether improvements from research will carry over to systems deployed for real-world use. In this work, we recommend three specific methods that are relatively easy to implement and result in much stronger experimental systems. Beyond reporting significantly higher BLEU scores, we conduct an in-depth analysis of where improvements originate and what inherent weaknesses of basic NMT models are being addressed. We then compare the relative gains afforded by several other techniques proposed in the literature when starting with vanilla systems versus our stronger baselines, showing that experimental conclusions may change depending on the baseline chosen. This indicates that choosing a strong baseline is crucial for reporting reliable experimental results.",
                authors=["Author 3", "Author 4"],
                categories=["cs.CL", "cs.LG"],
                published_date="2021-06-30",
                text_chunks=[
                    TextChunk(id="chunk4", type=ChunkType.TEXT, text="Interest in neural machine translation has grown rapidly as its effectiveness has been demonstrated."),
                    TextChunk(id="chunk5", type=ChunkType.TEXT, text="We recommend three specific methods that are relatively easy to implement and result in much stronger experimental systems."),
                    TextChunk(id="chunk6", type=ChunkType.TEXT, text="Choosing a strong baseline is crucial for reporting reliable experimental results.")
                ],
                figure_chunks=[],
                table_chunks=[],
                metadata={},
                pdf_path="dummy_path_2.pdf",
                HTML_path=None,
                comments='Test paper 2 - Neural Machine Translation'
            ),
            DocSet(
                doc_id="tinybert_2020",
                title="TinyBERT: Distilling BERT for Natural Language Understanding",
                abstract="Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks. However, pre-trained language models are usually computationally expensive, so it is difficult to efficiently execute them on resource-restricted devices. To accelerate inference and reduce model size while maintaining accuracy, we first propose a novel Transformer distillation method that is specially designed for knowledge distillation (KD) of the Transformer-based models. By leveraging this new KD method, the plenty of knowledge encoded in a large teacher BERT can be effectively transferred to a small student Tiny-BERT. Then, we introduce a new two-stage learning framework for TinyBERT, which performs Transformer distillation at both the pretraining and task-specific learning stages. This framework ensures that TinyBERT can capture the general-domain as well as the task-specific knowledge in BERT. TinyBERT with 4 layers is empirically effective and achieves more than 96.8% the performance of its teacher BERTBASE on GLUE benchmark, while being 7.5x smaller and 9.4x faster on inference. TinyBERT with 4 layers is also significantly better than 4-layer state-of-the-art baselines on BERT distillation, with only about 28% parameters and about 31% inference time of them. Moreover, TinyBERT with 6 layers performs on-par with its teacher BERTBASE",
                authors=["Xiaoqi Jiao", "Yichun Yin", "Lifeng Shang", "Xin Jiang", "Xiao Chen", "Linlin Li", "Fang Wang", "Qun Liu"],
                categories=["cs.CL", "cs.AI"],
                published_date="2020-09-15",
                text_chunks=[
                    TextChunk(id="chunk7", type=ChunkType.TEXT, text="Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks."),
                    TextChunk(id="chunk8", type=ChunkType.TEXT, text="We first propose a novel Transformer distillation method that is specially designed for knowledge distillation."),
                    TextChunk(id="chunk9", type=ChunkType.TEXT, text="TinyBERT with 4 layers achieves more than 96.8% the performance of its teacher BERTBASE on GLUE benchmark.")
                ],
                figure_chunks=[],
                table_chunks=[],
                metadata={},
                pdf_path="dummy_path_3.pdf",
                HTML_path=None,
                comments='Test paper 3 - TinyBERT'
            ),
            DocSet(
                doc_id="cv_cnn_2021",
                title="阿拉丁神灯",
                abstract="pizza dog 中文啦啦啦啦啦啦阿拉",
                authors=["Author 7", "Author 8"],
                categories=["cs.CV", "cs.AI"],
                published_date="2021-07-01",
                text_chunks=[
                    TextChunk(id="chunk10", type=ChunkType.TEXT, text="Deep learning approaches for computer vision tasks using convolutional neural networks have revolutionized the field."),
                    TextChunk(id="chunk11", type=ChunkType.TEXT, text="The hierarchical feature learning capability of CNNs allows them to automatically learn relevant features from raw pixel data."),
                    TextChunk(id="chunk12", type=ChunkType.TEXT, text="Recent advances include residual networks, attention mechanisms, and efficient architectures like MobileNets.")
                ],
                figure_chunks=[],
                table_chunks=[],
                metadata={},
                pdf_path="dummy_path_4.pdf",
                HTML_path=None,
                comments='Test paper 4 - Computer Vision'
            )
        ]
        
        self.logger.info(f"Created {len(self.test_papers)} test papers based on test_gritlm.py")
        return self.test_papers
    
    def initialize_databases(self, data: List[DocSet]) -> None:
        """初始化真实数据库
        
        Args:
            data: 测试论文数据列表
        """
        self.logger.info("Initializing GritLM databases...")
        
        # 初始化向量数据库
        vector_db_path = self.config['vector_db']['db_path']
        model_name = self.config['vector_db'].get('model_name', 'GritLM/GritLM-7B')
        vector_dim = self.config['vector_db'].get('vector_dim', 4096)
        
        # 清理现有向量数据库
        if os.path.exists(f"{vector_db_path}.index"):
            os.remove(f"{vector_db_path}.index")
        if os.path.exists(f"{vector_db_path}.entries"):
            os.remove(f"{vector_db_path}.entries")
        
        self.vector_db = VectorDB(
            db_path=vector_db_path,
            model_name=model_name,
            vector_dim=vector_dim
        )
        self.logger.info(f"GritLM vector database initialized: {vector_db_path}")
        
        # 初始化元数据数据库
        db_url = self.config['metadata_db']['db_url']
        self.engine = create_engine(db_url)
        
        # 重新创建表
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.logger.info("Metadata database tables recreated")
        
        self.metadata_db = MetadataDB(db_path=db_url)
        self.logger.info("Metadata database initialized")
        
        # 初始化PaperIndexer
        self.indexer = PaperIndexer(self.vector_db, self.metadata_db, None)
        self.logger.info("PaperIndexer initialized with GritLM databases")
        
        # 索引测试数据
        self.logger.info("Indexing test papers with GritLM...")
        
        indexing_results = self.indexer.index_papers(data, store_images=False)
        
        # 检查索引结果
        for doc_id, status in indexing_results.items():
            if not all(status.values()):
                failed_dbs = [db for db, success in status.items() if not success]
                self.logger.warning(f"Failed to index paper {doc_id} in databases: {failed_dbs}")
        
        self.logger.info("GritLM database initialization completed")
        self._display_storage_info()


    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human readable format.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted string with appropriate unit
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"


    def _display_storage_info(self) -> None:
        """Display current storage information for metadata_db and vector_db.
        
        Shows detailed information about database storage including file sizes,
        record counts, and database connectivity status.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("📊 CURRENT STORAGE INFORMATION")
            self.logger.info("=" * 60)
            
            # Vector DB Storage Information
            vector_db_path = self.config['vector_db']['db_path']
            self.logger.info(f"🗂️  Vector Database Path: {vector_db_path}")
            
            # Check if vector database exists and get info
            if os.path.exists(vector_db_path):
                index_file = os.path.join(vector_db_path, "index.pkl")
                faiss_file = os.path.join(vector_db_path, "index.faiss")
                
                if os.path.exists(index_file) and os.path.exists(faiss_file):
                    # Get file sizes
                    index_size = os.path.getsize(index_file)
                    faiss_size = os.path.getsize(faiss_file)
                    total_size = index_size + faiss_size
                    
                    self.logger.info(f"   ✅ Vector DB exists")
                    self.logger.info(f"   📁 Index file size: {self._format_bytes(index_size)}")
                    self.logger.info(f"   📁 FAISS file size: {self._format_bytes(faiss_size)}")
                    self.logger.info(f"   📊 Total size: {self._format_bytes(total_size)}")
                    
                    # Try to get vector count
                    try:
                        # First try to use initialized VectorDB object if available
                        if hasattr(self, 'vector_db') and self.vector_db and hasattr(self.vector_db, 'faiss_store'):
                            vector_count = len(self.vector_db.faiss_store.docstore._dict)
                            self.logger.info(f"   🔢 Vector count: {vector_count}")
                        else:
                            # Fallback: Quick check without full model loading
                            import pickle
                            with open(index_file, 'rb') as f:
                                index_data = pickle.load(f)
                                if hasattr(index_data, 'docstore') and hasattr(index_data.docstore, '_dict'):
                                    vector_count = len(index_data.docstore._dict)
                                    self.logger.info(f"   🔢 Vector count: {vector_count}")
                                else:
                                    self.logger.info(f"   🔢 Vector count: Unable to determine")
                    except Exception as e:
                        self.logger.info(f"   🔢 Vector count: Unable to determine ({str(e)})")
                else:
                    self.logger.info(f"   ❌ Vector DB files incomplete or missing")
            else:
                self.logger.info(f"   ❌ Vector DB directory does not exist")
            
            self.logger.info("")
            
            # Metadata DB Storage Information
            db_url = self.config['metadata_db']['db_url']
            self.logger.info(f"🗄️  Metadata Database URL: {db_url}")
            
            # Try to connect and get metadata info
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    # Check if tables exist
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    
                    if 'papers' in tables:
                        # Get paper count
                        result = conn.execute(text("SELECT COUNT(*) FROM papers"))
                        paper_count = result.scalar()
                        self.logger.info(f"   ✅ Metadata DB connected")
                        self.logger.info(f"   📄 Papers table exists")
                        self.logger.info(f"   🔢 Paper count: {paper_count}")
                        
                        # Get text chunks count if table exists
                        if 'text_chunks' in tables:
                            result = conn.execute(text("SELECT COUNT(*) FROM text_chunks"))
                            chunk_count = result.scalar()
                            self.logger.info(f"   📝 Text chunks count: {chunk_count}")
                        
                        # Get database size (PostgreSQL specific)
                        if 'postgresql' in db_url:
                            try:
                                result = conn.execute(text("""
                                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                                """))
                                db_size = result.scalar()
                                self.logger.info(f"   💾 Database size: {db_size}")
                            except Exception:
                                self.logger.info(f"   💾 Database size: Unable to determine")
                    else:
                        self.logger.info(f"   ❌ Papers table does not exist")
                        self.logger.info(f"   📋 Available tables: {', '.join(tables) if tables else 'None'}")
                        
            except Exception as e:
                self.logger.info(f"   ❌ Cannot connect to metadata DB: {str(e)}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Failed to display storage information: {str(e)}")

    
    def run_tests(self) -> Dict[str, Any]:
        """运行GritLM测试
        
        Returns:
            测试结果字典
        """
        self.logger.info("Running GritLM tests...")
        
        results = {
            'gritlm_indexing': self._test_gritlm_indexing(),
            'gritlm_vector_search': self._test_gritlm_vector_search(),
            #'gritlm_instruction_effectiveness': self._test_gritlm_instruction_effectiveness(),
            #'gritlm_similarity_ranking': self._test_gritlm_similarity_ranking(),
        }
        
        # 统计测试结果
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        
        self.logger.info(f"GritLM Test Results: {passed_tests}/{total_tests} tests passed")
        
        return results
    
    def _test_gritlm_indexing(self) -> Dict[str, Any]:
        """测试GritLM文档索引功能"""
        print("\n" + "="*60)
        print("🧪 TEST: _test_gritlm_indexing - 测试GritLM文档索引")
        print("="*60)
        try:
            # 检查向量数据库中的文档数量
            expected_count = len(self.test_papers)
            
            # 通过搜索验证文档是否被正确索引
            search_results = self.indexer.find_similar_papers(
                query="test query",
                top_k=expected_count * 2,  # 获取更多结果以确保找到所有文档
                search_strategies=[('vector', 0.1)]  # 使用很低的阈值
            )
            
            found_doc_ids = set()
            for result in search_results:
                if result.get('doc_id'):
                    found_doc_ids.add(result['doc_id'])
            
            expected_doc_ids = {paper.doc_id for paper in self.test_papers}
            
            # 验证所有文档都被找到
            all_docs_found = expected_doc_ids.issubset(found_doc_ids)
            success = all_docs_found and len(found_doc_ids) >= expected_count
            
            details = f"Indexed {len(found_doc_ids)} documents, expected {expected_count}. Found docs: {found_doc_ids}"
            
            self.log_test_result("GritLM Indexing", success, details)
            return {'success': success, 'found_count': len(found_doc_ids), 'expected_count': expected_count, 'details': details}
            
        except Exception as e:
            self.log_test_result("GritLM Indexing", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_gritlm_vector_search(self) -> Dict[str, Any]:
        """测试GritLM向量搜索功能"""
        print("\n" + "="*60)
        print("🧪 TEST: _test_gritlm_vector_search - 测试GritLM向量搜索")
        print("="*60)
        try:
            # 使用配置文件中的测试查询
            test_queries = ["Are there any research papers on methods to compress large-scale language models using task-agnostic knowledge distillation techniques?"]
            
            all_search_success = True
            search_results_summary = {}
            
            for query in test_queries:
                results = self.indexer.find_similar_papers(
                    query=query,
                    top_k=3,
                    search_strategies=[('vector', 0.8)],
                    result_include_types=['metadata','search_parameters']
                )

                for result in results:
                    print(result)
                
                search_success = len(results) > 0
                all_search_success = all_search_success and search_success
                
                search_results_summary[query] = {
                    'results_count': len(results),
                    'success': search_success,
                    'top_result': results[0].get('title', 'N/A') if results else 'N/A'
                }
                
                print(f"Query: '{query}' -> {len(results)} results, top: {search_results_summary[query]['top_result']}")
            
            success = all_search_success
            details = f"Tested {len(test_queries)} queries, all successful: {all_search_success}"
            
            self.log_test_result("GritLM Vector Search", success, details)
            return {'success': success, 'queries_tested': len(test_queries), 'search_summary': search_results_summary, 'details': details}
            
        except Exception as e:
            self.log_test_result("GritLM Vector Search", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_gritlm_instruction_effectiveness(self) -> Dict[str, Any]:
        """测试GritLM指令效果"""
        print("\n" + "="*60)
        print("🧪 TEST: _test_gritlm_instruction_effectiveness - 测试GritLM指令效果")
        print("="*60)
        try:
            # 测试查询
            query = "BERT knowledge distillation"
            
            # 获取当前使用的指令
            current_instruction = self.config.get('gritlm', {}).get('query_instruction', 
                "Given a scientific paper title, retrieve the paper's abstract")
            
            # 执行搜索（使用当前指令）
            results_with_instruction = self.indexer.find_similar_papers(
                query=query,
                top_k=3,
                search_strategies=[('vector', 0.8)],
                result_include_types=['metadata']
            )
            
            # 验证搜索结果质量
            success = len(results_with_instruction) > 0
            
            # 检查结果相关性（简单检查是否包含相关关键词）
            relevant_results = 0
            for result in results_with_instruction:
                title = result.get('title', '').lower()
                abstract = result.get('abstract', '').lower()
                if 'bert' in title or 'bert' in abstract or 'distill' in title or 'distill' in abstract:
                    relevant_results += 1
            
            relevance_score = relevant_results / len(results_with_instruction) if results_with_instruction else 0
            
            success = success and relevance_score > 0.3  # 至少30%的结果相关
            
            details = f"Query: '{query}', Results: {len(results_with_instruction)}, Relevant: {relevant_results}, Relevance: {relevance_score:.2f}"
            
            self.log_test_result("GritLM Instruction Effectiveness", success, details)
            return {
                'success': success, 
                'results_count': len(results_with_instruction),
                'relevant_results': relevant_results,
                'relevance_score': relevance_score,
                'details': details
            }
            
        except Exception as e:
            self.log_test_result("GritLM Instruction Effectiveness", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_gritlm_similarity_ranking(self) -> Dict[str, Any]:
        """测试GritLM相似度排序功能"""
        print("\n" + "="*60)
        print("🧪 TEST: _test_gritlm_similarity_ranking - 测试GritLM相似度排序")
        print("="*60)
        try:
            # 测试查询
            query = "transformer models"
            
            # 执行搜索
            results = self.indexer.find_similar_papers(
                query=query,
                top_k=4,  # 获取所有4个文档
                search_strategies=[('vector', 0.8)],
                result_include_types=['metadata', 'search_parameters']
            )
            
            success = len(results) > 0
            
            # 检查结果是否按相似度排序
            # 注意：这里我们假设search_parameters包含相似度分数
            # 如果没有，我们至少验证结果不为空
            ranking_valid = True
            if len(results) > 1:
                # 简单验证：检查结果数量是否合理
                ranking_valid = len(results) <= 4  # 不应该超过我们有的文档数量
            
            # 验证结果质量
            quality_score = 0
            for i, result in enumerate(results):
                title = result.get('title', '').lower()
                if 'transformer' in title or 'bert' in title or 'model' in title:
                    quality_score += 1
            
            quality_ratio = quality_score / len(results) if results else 0
            
            success = success and ranking_valid and quality_ratio > 0.2
            
            details = f"Query: '{query}', Results: {len(results)}, Quality ratio: {quality_ratio:.2f}, Ranking valid: {ranking_valid}"
            
            self.log_test_result("GritLM Similarity Ranking", success, details)
            return {
                'success': success,
                'results_count': len(results),
                'quality_ratio': quality_ratio,
                'ranking_valid': ranking_valid,
                'details': details
            }
            
        except Exception as e:
            self.log_test_result("GritLM Similarity Ranking", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    config_path = Path("/data3/guofang/AIgnite-Solutions/AIgnite/test/configs/gritlm_testbed_config.yaml")
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Please create the configuration file or specify a different path with -c")
        sys.exit(1)
    
    # 创建并运行测试床
    logger.info("Initializing GritLM TestBed...")
    testbed = GritLMTestBed(str(config_path))
    
    # 从配置文件中读取清理设置
    cleanup_before_test = testbed.config.get('environment', {}).get('cleanup_before_test', True)
    cleanup_after_test = testbed.config.get('environment', {}).get('cleanup_after_test', True)

    print(f"清理设置: 清理前={cleanup_before_test}, 清理后={cleanup_after_test}")
    
    # 执行测试，使用配置文件中的清理设置
    testbed.execute(clean_before_test=cleanup_before_test, clean_after_test=cleanup_after_test)
