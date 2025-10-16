#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GritLM专用测试床

继承自TestBed，专门用于GritLM功能测试。
使用test_gritlm.py中的4篇真实文章进行测试。
"""

from experiments.recommendation.testbed.base_testbed import TestBed
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
        
        # 365个真实文档ID（从用户提供的列表）
        """
        self.test_doc_ids = [
            '2510.11121v1_abstract', '2510.11478v1_abstract', '2510.11183v1_abstract', '2510.11194v1_abstract', 
            '2510.10870v1_abstract', '2510.11036v1_abstract', '2510.11305v1_abstract', '2510.11152v1_abstract', 
            '2510.10910v1_abstract', '2510.11242v1_abstract', '2510.11166v1_abstract', '2510.10963v1_abstract', 
            '2510.11474v1_abstract', '2510.10980v1_abstract', '2510.11218v1_abstract', '2510.10989v1_abstract', 
            '2510.10895v1_abstract', '2510.10885v1_abstract', '2510.11062v2_abstract', '2510.11173v1_abstract', 
            '2510.10899v1_abstract', '2510.11287v1_abstract', '2510.11004v1_abstract', '2510.11260v1_abstract', 
            '2510.11168v1_abstract', '2510.10944v1_abstract', '2510.10878v1_abstract', '2510.11483v1_abstract', 
            '2510.10965v1_abstract', '2510.11306v1_abstract', '2510.11019v1_abstract', '2510.11041v1_abstract', 
            '2510.11421v1_abstract', '2510.11253v1_abstract', '2510.11116v1_abstract', '2510.11502v1_abstract', 
            '2510.11495v1_abstract', '2510.11108v1_abstract', '2510.11122v1_abstract', '2510.11115v1_abstract', 
            '2510.11131v1_abstract', '2510.11408v1_abstract', '2510.11225v1_abstract', '2510.11369v1_abstract', 
            '2510.11246v1_abstract', '2510.11058v1_abstract', '2510.11472v1_abstract', '2510.10977v1_abstract', 
            '2510.11286v1_abstract', '2510.11221v1_abstract', '2510.10872v1_abstract', '2510.11112v1_abstract', 
            '2510.11380v1_abstract', '2510.10921v1_abstract', '2510.11264v1_abstract', '2510.10959v1_abstract', 
            '2510.10942v1_abstract', '2510.11014v1_abstract', '2510.11456v1_abstract', '2510.11334v1_abstract', 
            '2510.11047v1_abstract', '2510.11040v1_abstract', '2510.10982v1_abstract', '2510.11065v1_abstract', 
            '2510.10994v1_abstract', '2510.10914v1_abstract', '2510.11434v1_abstract', '2510.11328v1_abstract', 
            '2510.11020v1_abstract', '2510.10877v1_abstract', '2510.10894v1_abstract', '2510.10998v1_abstract', 
            '2510.11169v1_abstract', '2510.11080v1_abstract', '2510.11354v1_abstract', '2510.11344v1_abstract', 
            '2510.10943v1_abstract', '2510.11107v1_abstract', '2510.10868v1_abstract', '2510.11085v1_abstract', 
            '2510.11083v1_abstract', '2510.11299v1_abstract', '2510.11347v1_abstract', '2510.11063v1_abstract', 
            '2510.10901v1_abstract', '2510.11340v2_abstract', '2510.11278v1_abstract', '2510.11390v1_abstract', 
            '2510.11210v1_abstract', '2510.10937v1_abstract', '2510.10990v1_abstract', '2510.10956v1_abstract', 
            '2510.11223v1_abstract', '2510.11171v1_abstract', '2510.11198v1_abstract', '2510.10869v1_abstract', 
            '2510.10960v1_abstract', '2510.10971v1_abstract', '2510.11189v1_abstract', '2510.11320v1_abstract', 
            '2510.11447v1_abstract', '2510.11274v1_abstract', '2510.11043v1_abstract', '2510.11393v1_abstract', 
            '2510.11124v1_abstract', '2510.11094v1_abstract', '2510.11283v1_abstract', '2510.11000v1_abstract', 
            '2510.10986v1_abstract', '2510.11394v1_abstract', '2510.11269v1_abstract', '2510.11181v1_abstract', 
            '2510.11176v1_abstract', '2510.11204v1_abstract', '2510.11310v1_abstract', '2510.11144v1_abstract', 
            '2510.11143v1_abstract', '2510.10936v1_abstract', '2510.11388v1_abstract', '2510.10962v1_abstract', 
            '2510.11164v1_abstract', '2510.11031v1_abstract', '2510.10975v2_abstract', '2510.11196v1_abstract', 
            '2510.11059v1_abstract', '2510.11084v1_abstract', '2510.10902v1_abstract', '2510.11313v1_abstract', 
            '2510.11103v1_abstract', '2510.10974v1_abstract', '2510.11307v1_abstract', '2510.11023v1_abstract', 
            '2510.10978v1_abstract', '2510.11391v1_abstract', '2510.11035v1_abstract', '2510.11005v1_abstract', 
            '2510.10931v1_abstract', '2510.10879v1_abstract', '2510.11217v1_abstract', '2510.11491v1_abstract', 
            '2510.11053v1_abstract', '2510.11151v1_abstract', '2510.11104v1_abstract', '2510.11498v1_abstract', 
            '2510.11275v1_abstract', '2510.11211v1_abstract', '2510.11296v1_abstract', '2510.11203v1_abstract', 
            '2510.10933v1_abstract', '2510.11259v1_abstract', '2510.11303v1_abstract', '2510.10903v1_abstract', 
            '2510.11224v1_abstract', '2510.10981v1_abstract', '2510.10915v1_abstract', '2510.11445v1_abstract', 
            '2510.11039v1_abstract', '2510.11106v1_abstract', '2510.11503v1_abstract', '2510.10952v1_abstract', 
            '2510.10961v1_abstract', '2510.11091v1_abstract', '2510.11026v1_abstract', '2510.11192v1_abstract', 
            '2510.11147v1_abstract', '2510.11110v1_abstract', '2510.11335v1_abstract', '2510.11007v1_abstract', 
            '2510.11400v1_abstract', '2510.10976v1_abstract', '2510.11444v1_abstract', '2510.11050v1_abstract', 
            '2510.10876v1_abstract', '2510.11417v1_abstract', '2510.11402v1_abstract', '2510.10912v1_abstract', 
            '2510.11300v1_abstract', '2510.11321v1_abstract', '2510.11001v1_abstract', '2510.10938v1_abstract', 
            '2510.11449v1_abstract', '2510.10988v1_abstract', '2510.11423v1_abstract', '2510.11471v1_abstract', 
            '2510.11016v1_abstract', '2510.11500v1_abstract', '2510.11413v1_abstract', '2510.11419v2_abstract', 
            '2510.11266v1_abstract', '2510.11090v1_abstract', '2510.11137v1_abstract', '2510.11386v2_abstract', 
            '2510.11341v1_abstract', '2510.11358v1_abstract', '2510.11368v1_abstract', '2510.10930v1_abstract', 
            '2510.11068v1_abstract', '2510.11153v1_abstract', '2510.11409v1_abstract', '2510.11089v1_abstract', 
            '2510.10929v1_abstract', '2510.11301v1_abstract', '2510.11339v1_abstract', '2510.11291v1_abstract', 
            '2510.11268v1_abstract', '2510.10913v1_abstract', '2510.11129v1_abstract', '2510.11095v1_abstract', 
            '2510.11418v1_abstract', '2510.10889v1_abstract', '2510.11295v1_abstract', '2510.10991v1_abstract', 
            '2510.10866v1_abstract', '2510.11170v1_abstract', '2510.11758v1_abstract', '2510.11243v1_abstract', 
            '2510.11076v1_abstract', '2510.11082v1_abstract', '2510.11049v1_abstract', '2510.10969v1_abstract', 
            '2510.11499v1_abstract', '2510.11448v1_abstract', '2510.11098v1_abstract', '2510.11182v1_abstract', 
            '2510.11442v1_abstract', '2510.11109v1_abstract', '2510.11212v1_abstract', '2510.11330v1_abstract', 
            '2510.11501v1_abstract', '2510.11251v1_abstract', '2510.11462v1_abstract', '2510.11258v1_abstract', 
            '2510.11411v2_abstract', '2510.11092v1_abstract', '2510.11389v1_abstract', '2510.11015v1_abstract', 
            '2510.11066v1_abstract', '2510.10987v1_abstract', '2510.11250v1_abstract', '2510.10951v1_abstract', 
            '2510.10893v1_abstract', '2510.11370v1_abstract', '2510.11028v1_abstract', '2510.10927v1_abstract', 
            '2510.10968v1_abstract', '2510.11255v1_abstract', '2510.11236v1_abstract', '2510.11245v1_abstract', 
            '2510.11473v1_abstract', '2510.10909v1_abstract', '2510.11175v1_abstract', '2510.11482v1_abstract', 
            '2510.10887v1_abstract', '2510.11209v1_abstract', '2510.10864v1_abstract', '2510.11148v1_abstract', 
            '2510.11056v1_abstract', '2510.10955v1_abstract', '2510.11100v1_abstract', '2510.11288v1_abstract', 
            '2510.10890v1_abstract', '2510.11017v1_abstract', '2510.11496v2_abstract', '2510.11293v1_abstract', 
            '2510.11316v1_abstract', '2510.10925v1_abstract', '2510.11012v1_abstract', '2510.11323v1_abstract', 
            '2510.11227v1_abstract', '2510.11453v1_abstract', '2510.11232v1_abstract', '2510.11195v1_abstract', 
            '2510.11302v1_abstract', '2510.10973v1_abstract', '2510.11162v1_abstract', '2510.11361v1_abstract', 
            '2510.11167v1_abstract', '2510.11237v1_abstract', '2510.10993v1_abstract', '2510.11410v2_abstract', 
            '2510.10984v1_abstract', '2510.11142v1_abstract', '2510.11414v1_abstract', '2510.11760v1_abstract', 
            '2510.11222v1_abstract', '2510.11190v2_abstract', '2510.11072v1_abstract', '2510.11140v1_abstract', 
            '2510.10966v1_abstract', '2510.10865v1_abstract', '2510.11234v1_abstract', '2510.11178v1_abstract', 
            '2510.11027v1_abstract', '2510.11018v1_abstract', '2510.11387v1_abstract', '2510.11439v1_abstract', 
            '2510.11087v1_abstract', '2510.10995v1_abstract', '2510.11073v1_abstract', '2510.11238v1_abstract', 
            '2510.11184v1_abstract', '2510.11317v1_abstract', '2510.11277v1_abstract', '2510.10964v1_abstract', 
            '2510.11133v1_abstract', '2510.11476v1_abstract', '2510.11475v1_abstract', '2510.11011v1_abstract', 
            '2510.11379v1_abstract', '2510.11079v1_abstract', '2510.11314v1_abstract', '2510.11057v1_abstract', 
            '2510.10880v1_abstract', '2510.10918v1_abstract', '2510.11416v1_abstract', '2510.11438v1_abstract', 
            '2510.11290v1_abstract', '2510.11188v1_abstract', '2510.11318v1_abstract', '2510.11282v1_abstract', 
            '2510.11360v1_abstract', '2510.11372v1_abstract', '2510.11398v1_abstract', '2510.11484v1_abstract', 
            '2510.10948v1_abstract', '2510.11343v1_abstract', '2510.11405v1_abstract', '2510.11003v1_abstract', 
            '2510.11119v1_abstract', '2510.11407v1_abstract', '2510.11308v1_abstract', '2510.10979v1_abstract', 
            '2510.11233v1_abstract', '2510.11179v1_abstract', '2510.11257v1_abstract', '2510.11174v1_abstract', 
            '2510.11276v1_abstract', '2510.11345v1_abstract', '2510.10920v1_abstract', '2510.10940v1_abstract', 
            '2510.10932v1_abstract', '2510.11254v1_abstract', '2510.11420v1_abstract', '2510.11160v1_abstract', 
            '2510.11052v1_abstract', '2510.11292v1_abstract', '2510.11117v1_abstract', '2510.11297v1_abstract', 
            '2510.11138v1_abstract', '2510.11096v1_abstract', '2510.11759v1_abstract', '2510.10862v1_abstract', 
            '2510.11235v1_abstract', '2510.11457v1_abstract', '2510.11374v1_abstract', '2510.10947v1_abstract', 
            '2510.11128v1_abstract', '2510.11454v1_abstract', '2510.11185v1_abstract', '2510.11401v1_abstract', 
            '2510.10886v1_abstract', '2510.11281v1_abstract', '2510.11346v1_abstract', '2510.11064v1_abstract', 
            '2510.11202v1_abstract', '2510.11141v1_abstract', '2510.11214v1_abstract', '2510.10892v1_abstract', 
            '2510.11331v1_abstract', '2510.11123v1_abstract'
        ]
        """
        self.test_doc_ids = [
            '2510.11079v1_abstract',
            '2510.11121v1_abstract', '2510.11478v1_abstract', '2510.11183v1_abstract', '2510.11194v1_abstract', 
            '2510.10870v1_abstract', '2510.11036v1_abstract', '2510.11305v1_abstract', '2510.11152v1_abstract', 
            '2510.10910v1_abstract', '2510.11242v1_abstract', '2510.11166v1_abstract', '2510.10963v1_abstract', ]
    
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
        """加载测试数据 - 从生产数据库读取真实文档
        
        Returns:
            测试论文数据列表
        """
        self.logger.info("Loading test papers from production database...")
        
        # 连接到生产数据库
        production_db_url = "postgresql://postgres:11111@localhost:5432/paperignition"
        production_metadata_db = MetadataDB(db_path=production_db_url)
        
        self.test_papers = []
        skipped_count = 0
        
        for doc_id_with_suffix in self.test_doc_ids:
            # 去除_abstract后缀
            doc_id = doc_id_with_suffix.replace('_abstract', '')
            
            try:
                # 从数据库获取元数据
                metadata = production_metadata_db.get_metadata(doc_id)
                
                if metadata is None:
                    self.logger.warning(f"Document not found in database: {doc_id}")
                    skipped_count += 1
                    continue
                
                # 创建DocSet对象（text_chunks保持空白，只使用abstract）
                docset = DocSet(
                    doc_id=metadata.get('doc_id', doc_id),
                    title=metadata.get('title', ''),
                    abstract=metadata.get('abstract', ''),
                    authors=metadata.get('authors') or [],
                    categories=metadata.get('categories') or [],
                    published_date=metadata.get('published_date', ''),
                    text_chunks=[],  # 空列表 - 只使用abstract
                    figure_chunks=[],
                    table_chunks=[],
                    metadata={},
                    pdf_path=metadata.get('pdf_path', ''),
                    HTML_path=metadata.get('HTML_path'),
                    comments=metadata.get('comments')
                )
                
                self.test_papers.append(docset)
                
            except Exception as e:
                self.logger.error(f"Error loading document {doc_id}: {str(e)}")
                skipped_count += 1
                continue
        
        self.logger.info(f"Loaded {len(self.test_papers)} test papers from production database, skipped {skipped_count}")
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
        
        '''
        indexing_results = self.indexer.index_papers(data, store_images=False)
        
        # 检查索引结果
        for doc_id, status in indexing_results.items():
            if not all(status.values()):
                failed_dbs = [db for db, success in status.items() if not success]
                self.logger.warning(f"Failed to index paper {doc_id} in databases: {failed_dbs}")
        '''
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
            #'gritlm_indexing': self._test_gritlm_indexing(),
            'gritlm_save_vectors': self._test_gritlm_save_vectors(),
            'gritlm_vector_search': self._test_gritlm_vector_search(),
            #'gritlm_instruction_effectiveness': self._test_gritlm_instruction_effectiveness(),
            #'gritlm_similarity_ranking': self._test_gritlm_similarity_ranking(),
        }
        
        # 统计测试结果
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        
        self.logger.info(f"GritLM Test Results: {passed_tests}/{total_tests} tests passed")
        
        return results

    def _test_gritlm_save_vectors(self) -> Dict[str, Any]:
        """测试GritLM向量保存功能"""
        print("\n" + "="*60)
        print("🧪 TEST: _test_gritlm_save_vectors - 测试GritLM向量保存功能")
        print("="*60)
        try:
            # 保存向量
            self.indexer.save_vectors(self.test_papers)
        except Exception as e:
            self.log_test_result("GritLM Vector Saving", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
        return {'success': True, 'details': "GritLM向量保存成功"}
    
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
                query="reinforcement learning",
                top_k=5,  # 获取更多结果以确保找到所有文档
                search_strategies=[('vector', 2.0)]  # 使用很低的阈值
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
            test_queries = ["Argumentation-Based Explainability for Legal AI"]
            
            all_search_success = True
            search_results_summary = {}
            
            for query in test_queries:
                results = self.indexer.find_similar_papers(
                    query=query,
                    top_k=3,
                    search_strategies=[('vector', 1.5)],
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
