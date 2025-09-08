"""Database modules for AIgnite."""
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, String, Integer, JSON, Text, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declared_attr
from sqlalchemy import text, Index
from sqlalchemy.sql import func
import logging
import os
from ..data.docset import DocSet, BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

Base = declarative_base()

class TableSchema(Base):
    """Database table schema for storing paper metadata and chunk IDs.
    Inherits structure from DocSet but only stores chunk IDs."""
    
    __tablename__ = 'papers'
    
    # SQLAlchemy columns
    id = Column(Integer, primary_key=True)
    doc_id = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    abstract = Column(Text)
    authors = Column(JSON)
    categories = Column(JSON)
    published_date = Column(String)
    pdf_data = Column(LargeBinary)
    chunk_ids = Column(JSON)  # Store text chunk IDs
    figure_ids = Column(JSON)  # Store figure chunk IDs
    table_ids = Column(JSON)  # Store table chunk IDs
    extra_metadata = Column(JSON)  # Store metadata dict
    pdf_path = Column(String)
    HTML_path = Column(String, nullable=True)
    blog = Column(Text, nullable=True)  # New field for long text blog
    comments = Column(Text, nullable=True)  # Store comments field
    
    # Add tsvector column for full-text search
    __table_args__ = (
        Index(
            'idx_fts',
            text('to_tsvector(\'english\', coalesce(title, \'\') || \' \' || coalesce(abstract, \'\'))'),
            postgresql_using='gin'
        ),
    )
    
    @classmethod
    def from_docset(cls, docset: DocSet, pdf_data: bytes = None) -> 'TableSchema':
        """Create a TableSchema instance from a DocSet object.
        
        Args:
            docset: DocSet object containing paper information
            pdf_data: Optional PDF binary data
            
        Returns:
            TableSchema instance initialized with DocSet data
        """
        return cls(
            doc_id=docset.doc_id,
            title=docset.title,
            abstract=docset.abstract,
            authors=docset.authors,
            categories=docset.categories,
            published_date=docset.published_date,
            pdf_data=pdf_data,
            chunk_ids=[chunk.id for chunk in docset.text_chunks],
            figure_ids=[chunk.id for chunk in docset.figure_chunks],
            table_ids=[chunk.id for chunk in docset.table_chunks],
            extra_metadata=docset.metadata,
            pdf_path=docset.pdf_path,
            HTML_path=docset.HTML_path,
            blog=getattr(docset, 'blog', None),  # Support blog field if present
            comments=docset.comments  # Store comments field
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TableSchema instance to dictionary format.
        
        Returns:
            Dictionary containing paper metadata
        """
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'categories': self.categories,
            'published_date': self.published_date,
            'chunk_ids': self.chunk_ids,
            'figure_ids': self.figure_ids,
            'table_ids': self.table_ids,
            'metadata': self.extra_metadata,
            'blog': self.blog,  # Include blog field in output
            'comments': self.comments,  # Include comments field in output
            'pdf_path': self.pdf_path,  # Add pdf_path
            'HTML_path': self.HTML_path  # Add HTML_path
        }

class TextChunkRecord(Base):
    """Database table schema for storing text chunk content.
    This table stores the actual text content of each chunk with proper ordering."""
    
    __tablename__ = 'text_chunks'
    
    # SQLAlchemy columns
    id = Column(String, primary_key=True)  # Primary key: doc_id + chunk_id
    doc_id = Column(String, nullable=False)  # Associated paper ID
    chunk_id = Column(String, nullable=False)  # Original chunk ID within document
    text_content = Column(Text, nullable=False)  # Actual text content
    chunk_order = Column(Integer, nullable=False)  # Order within document for sorting
    created_at = Column(DateTime, default=func.now())  # Creation timestamp
    
    # Add tsvector column for full-text search on chunk content
    __table_args__ = (
        Index('idx_chunk_doc_id', 'doc_id'),  # Index for document queries
        Index('idx_chunk_order', 'doc_id', 'chunk_order'),  # Index for ordering
        Index('idx_chunk_text_fts', text("to_tsvector('english', text_content)"), postgresql_using='gin'),  # Full-text search index
        Index('idx_chunk_doc_chunk', 'doc_id', 'chunk_id', unique=True),  # Unique constraint
    )

class MetadataDB:
    def __init__(self, db_path: str = None):
        """Initialize database connection.
        
        Args:
            db_path: Database connection string
            
        Raises:
            ValueError: If db_path is not provided
        """
        if not db_path:
            raise ValueError("Database path must be provided for MetadataDB initialization")
            
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create full-text search function if using PostgreSQL
        self._create_fts_function()

    def _create_fts_function(self):
        """Create custom full-text search ranking function."""
        session = self.Session()
        try:
            # Create a custom ranking function that combines ts_rank_cd with document weights
            session.execute(text("""
                CREATE OR REPLACE FUNCTION fts_rank(
                    title text,
                    abstract text,
                    q tsquery,
                    title_weight float DEFAULT 0.7,
                    abstract_weight float DEFAULT 0.3
                ) RETURNS float AS $$
                BEGIN
                    RETURN (
                        title_weight * ts_rank_cd(
                            setweight(to_tsvector('english', coalesce(title, '')), 'A'),
                            q
                        ) +
                        abstract_weight * ts_rank_cd(
                            setweight(to_tsvector('english', coalesce(abstract, '')), 'B'),
                            q
                        )
                    );
                END;
                $$ LANGUAGE plpgsql;
            """))
            session.commit()
        except Exception as e:
            session.rollback()
            logging.warning(f"Failed to create FTS function: {str(e)}")
        finally:
            session.close()

    # Original search method (commented out, kept for backup)
    # def search_papers(
    #     self,
    #     query: str,
    #     top_k: int = 10,
    #     similarity_cutoff: float = 0.1,
    #     filters: Optional[Dict[str, Any]] = None
    # ) -> List[Dict[str, Any]]:
    #     """Search papers using PostgreSQL full-text search."""
    #     session = self.Session()
    #     try:
    #         # First, let's debug the query parsing
    #         debug_query = session.execute(text("""
    #             SELECT plainto_tsquery('english', :query) as parsed_query
    #         """), {'query': query}).scalar()
    #         logger.debug(f"Parsed query: {debug_query}")
    #         # Use OR (|) between words for more forgiving matching
    #         or_query = ' | '.join(query.split())

    #         # Build WHERE clause for filters
    #         where_clause = "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, '')) @@ to_tsquery('english', :query)"
    #         filter_params = {'query': or_query, 'cutoff': similarity_cutoff, 'limit': top_k}
            
    #         # Handle new filter structure
    #         if filters:
    #             if "include" in filters or "exclude" in filters:
    #                 # New filter structure
    #                 from ..index.filter_parser import FilterParser
    #         filter_parser = FilterParser()
    #         filter_where, filter_params_update = filter_parser.get_sql_conditions(filters)
                    
    #         if filter_where != "1=1":
    #             where_clause += f" AND ({filter_where})"
                    
    #         # Update parameters
    #         for key, value in filter_params_update.items():
    #             filter_params[f"filter_{key}"] = value
    #         # Modified search query with to_tsquery for OR logic
    #         search_results = session.execute(text(f"""
    #         WITH search_results AS (
    #             SELECT
    #             doc_id,
    #             title,
    #             abstract,
    #             authors,
    #             categories,
    #             published_date,
    #             extra_metadata as metadata,
    #             fts_rank(
    #             title,
    #             abstract,
    #             to_tsquery('english', :query)
    #         ) as score,
    #         ts_headline(
    #             'english',
    #             coalesce(title, '') || ' ' || coalesce(abstract, ''),
    #             to_tsquery('english', :query),
    #             'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20'
    #         ) as headline
    #     FROM papers
    #     WHERE {where_clause}
    # )
    # SELECT *
    # FROM search_results
    # WHERE score >= :cutoff
    # ORDER BY score DESC
    # LIMIT :limit
    # """), filter_params)
    #         # Debug the results
    #         results = []
    #         for row in search_results:
    #             result_dict = {
    #             'doc_id': row.doc_id,
    #             'score': float(row.score),
    #             'metadata': {
    #             'title': row.title,
    #             'abstract': row.abstract,
    #             'authors': row.authors,
    #             'categories': row.categories,
    #             'published_date': row.published_date,
    #             **(row.metadata or {})
    #         },
    #         'matched_text': row.headline
    #     }
    #     results.append(result_dict)
    #         logger.debug(f"Found result: {result_dict['doc_id']} with score {result_dict['score']}")

    #         return results

    #     except Exception as e:
    #         logger.error(f"Search failed: {str(e)}")
    #         return []
    #     finally:
    #         session.close()

    def save_paper(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any], text_chunks: Optional[List] = None) -> bool:
        """Save paper PDF and metadata.
        Args:
            doc_id: Document ID
            pdf_path: Path to PDF file
            metadata: Dictionary containing paper metadata with required fields:
                     title, abstract, authors, categories, published_date
            text_chunks: Optional list of text chunks to save
            
        Returns:
            True if successful, False if doc_id already exists or on error
            
        Raises:
            ValueError: If required metadata fields are missing
        """
        required_fields = ['title', 'abstract', 'authors', 'categories', 'published_date']
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {', '.join(missing_fields)}")
            
        session = self.Session()
        try:
            # Check if paper already exists
            if session.query(TableSchema).filter_by(doc_id=doc_id).first():
                logging.warning(f"Paper {doc_id} already exists. Skip saving.")
                return False

            # Handle PDF data - if pdf_path is None, set pdf_data to None
            pdf_data = None
            if pdf_path:
                try:
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                except Exception as e:
                    logging.warning(f"Failed to read PDF file {pdf_path}: {str(e)}")
                    pdf_data = None

            # Create new paper
            paper = TableSchema(
                doc_id=doc_id,
                title=metadata['title'],
                abstract=metadata['abstract'],
                authors=metadata['authors'],
                categories=metadata['categories'],
                published_date=metadata['published_date'],
                pdf_data=pdf_data,
                chunk_ids=metadata.get('chunk_ids', []),
                figure_ids=metadata.get('figure_ids', []),
                table_ids=metadata.get('table_ids', []),
                extra_metadata=metadata.get('metadata', {}),
                pdf_path=pdf_path,
                HTML_path=metadata.get('HTML_path'),
                blog=metadata.get('blog', None),  # Support blog field if present
                comments=metadata.get('comments', '')  # Store comments field
            )
            session.add(paper)
            
            # Save text chunks content if available
            if text_chunks:
                chunk_success = self._save_text_chunks(session, doc_id, text_chunks)
                if not chunk_success:
                    session.rollback()
                    return False
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to save paper {doc_id}: {str(e)}")
            return False
        finally:
            session.close()

    def _save_text_chunks(self, session, doc_id: str, text_chunks: List) -> bool:
        """Save text chunk content to database using doc_id + chunk_id as unique identifier.
        
        Args:
            session: Database session
            doc_id: Document ID
            text_chunks: List of text chunks from DocSet
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for chunk in text_chunks:
                # Create unique ID: doc_id + chunk_id
                unique_chunk_id = f"{doc_id}_{chunk.id}"
                
                # Extract order number from chunk_id for sorting
                chunk_order = self._extract_order(chunk.id)
                
                chunk_record = TextChunkRecord(
                    id=unique_chunk_id,        # Primary key: doc_id + chunk_id
                    doc_id=doc_id,             # Document ID
                    chunk_id=chunk.id,         # Original chunk_id
                    text_content=chunk.text,   # Text content
                    chunk_order=chunk_order    # Order for sorting
                )
                session.add(chunk_record)
            return True
        except Exception as e:
            logger.error(f"Failed to save text chunks for {doc_id}: {str(e)}")
            return False

    def _extract_order(self, chunk_id: str) -> int:
        """Extract order number from chunk_id for sorting.
        
        Args:
            chunk_id: Chunk ID string (e.g., "chunk_001", "text_1")
            
        Returns:
            Order number as integer, 0 if no number found
        """
        import re
        match = re.search(r'(\d+)$', chunk_id)
        return int(match.group(1)) if match else 0

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        session = self.Session()
        try:
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not paper:
                return None
            return paper.to_dict()
        except Exception as e:
            logging.error(f"Failed to get metadata for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close()

    def get_pdf(self, doc_id: str, save_path: str = None) -> Optional[bytes]:
        """Retrieve PDF data for a document.
        
        Args:
            doc_id: Document ID
            save_path: Optional path to save the PDF file
            
        Returns:
            PDF binary data if save_path is None, else None
        """
        session = self.Session()
        try:
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not paper or not paper.pdf_data:
                return None

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(paper.pdf_data)
                return None
            return paper.pdf_data
        except Exception as e:
            logging.error(f"Failed to get PDF for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close()

    def delete_paper(self, doc_id: str) -> bool:
        """Delete paper and its metadata, including all text chunks.
        
        Args:
            doc_id: Document ID
            
        Returns:
            paper_metadata_deleted(True/False),text_chunks_deleted(True/False)
        """
        session = self.Session()
        text_chunks_deleted = False
        paper_metadata_deleted = False
        try:
            # Delete text chunks first (due to foreign key constraint)
            text_chunks_deleted = session.query(TextChunkRecord)\
                .filter_by(doc_id=doc_id)\
                .delete()
            if text_chunks_deleted > 0:
                text_chunks_deleted = True
            logger.debug(f"Deleted {text_chunks_deleted} text chunks for doc_id {doc_id}")
            
            # Delete the paper
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if paper:
                paper_metadata_deleted = True
            
            session.delete(paper)
            session.commit()
            return paper_metadata_deleted,text_chunks_deleted

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to delete paper {doc_id}: {str(e)}")
            return False
        finally:
            session.close()

    def save_blog(self, doc_id: str, blog: str) -> bool:
        """Save or update the blog text for a given document ID.
        
        Args:
            doc_id: Document ID
            blog: Blog text to save
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not paper:
                return False
            paper.blog = blog
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to save blog for doc_id {doc_id}: {str(e)}")
            return False
        finally:
            session.close()

    def get_blog(self, doc_id: str) -> Optional[str]:
        """Retrieve the blog text for a given document ID.
        
        Args:
            doc_id: Document ID
        Returns:
            Blog text if found, None otherwise
        """
        session = self.Session()
        try:
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not paper:
                return None
            return paper.blog
        except Exception as e:
            logging.error(f"Failed to get blog for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close()

    def get_text_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all text chunks for a document, ordered by chunk_order.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of text chunks with metadata, ordered by chunk_order
        """
        session = self.Session()
        try:
            chunks = session.query(TextChunkRecord)\
                .filter_by(doc_id=doc_id)\
                .order_by(TextChunkRecord.chunk_order)\
                .all()
            
            return [
                {
                    'chunk_id': chunk.chunk_id,
                    'text_content': chunk.text_content,
                    'chunk_order': chunk.chunk_order,
                    'created_at': chunk.created_at
                }
                for chunk in chunks
            ]
        except Exception as e:
            logger.error(f"Failed to get text chunks for doc_id {doc_id}: {str(e)}")
            return []
        finally:
            session.close()

    def get_full_text(self, doc_id: str) -> Optional[str]:
        """Retrieve the complete text content of a document by concatenating all text chunks.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Complete text content as string, or None if not found
        """
        chunks = self.get_text_chunks(doc_id)
        if not chunks:
            return None
        
        # Concatenate chunks in order with proper spacing
        return '\n\n'.join([chunk['text_content'] for chunk in chunks])

    def search_in_chunks(self, query: str, top_k: int = 10, similarity_cutoff: float = 0.1) -> List[Dict[str, Any]]:
        """Search within text chunk content using PostgreSQL full-text search.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            similarity_cutoff: Minimum similarity score threshold
            
        Returns:
            List of search results with chunk information
        """
        session = self.Session()
        try:
            # Use OR logic between words for more forgiving matching
            or_query = ' | '.join(query.split())
            
            search_results = session.execute(text("""
                SELECT
                    tc.doc_id,
                    tc.chunk_id,
                    tc.text_content,
                    tc.chunk_order,
                    ts_rank(
                        to_tsvector('english', tc.text_content),
                        to_tsquery('english', :query)
                    ) as score,
                    ts_headline(
                        'english',
                        tc.text_content,
                        to_tsquery('english', :query),
                        'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20'
                    ) as matched_text
                FROM text_chunks tc
                WHERE to_tsvector('english', tc.text_content) @@ to_tsquery('english', :query)
                ORDER BY score DESC
                LIMIT :limit
            """), {'query': or_query, 'limit': top_k})
            
            results = []
            for row in search_results:
                if float(row.score) >= similarity_cutoff:
                    results.append({
                        'doc_id': row.doc_id,
                        'chunk_id': row.chunk_id,
                        'chunk_order': row.chunk_order,
                        'text_content': row.text_content,
                        'score': float(row.score),
                        'matched_text': row.matched_text
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk search failed: {str(e)}")
            return []
        finally:
            session.close()

    def get_filtered_doc_ids(self, filters: Optional[Dict[str, Any]]) -> List[str]:
        """Apply filters first to get candidate document ID list.
        
        Args:
            filters: Filter dictionary with include/exclude conditions
            
        Returns:
            List of document IDs that pass the filters
        """
        if not filters:
            return []
        
        session = self.Session()
        try:
            from ..index.filter_parser import FilterParser
            filter_parser = FilterParser()
            filter_where, filter_params = filter_parser.get_sql_conditions(filters)
            
            if filter_where == "1=1":
                return []
            
            # Only get doc_ids, no full-text search
            result = session.execute(text(f"""
                SELECT doc_id FROM papers WHERE {filter_where}
            """), filter_params)
            
            return [row.doc_id for row in result]
            
        except Exception as e:
            logger.error(f"Filter application failed: {str(e)}")
            return []
        finally:
            session.close()

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        similarity_cutoff: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Search papers using PostgreSQL full-text search with filter-first approach.
        
        Args:
            query: Search query string
            filters: Optional filters to apply first
            top_k: Maximum number of results to return
            similarity_cutoff: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata and matched text
        """
        session = self.Session()
        try:
            # Step 1: Apply filters to get candidate doc_ids (if filters provided)
            if filters:
                candidate_doc_ids = self.get_filtered_doc_ids(filters)
                
                if not candidate_doc_ids:
                    logger.info("No documents match the filters")
                    return []
                
                logger.info(f"Filter applied: {len(candidate_doc_ids)} candidate documents")
                # Step 2: Perform full-text search on filtered results
            else:
                candidate_doc_ids = None
                logger.info("No filters provided, searching all documents")
                # Step 2: Perform full-text search on all documents
            or_query = ' | '.join(query.split())
            
            # Build WHERE clause based on whether filters are applied
            if candidate_doc_ids:
                where_clause = "doc_id = ANY(:candidate_doc_ids) AND to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, '')) @@ to_tsquery('english', :query)"
                query_params = {
                    'query': or_query,
                    'candidate_doc_ids': candidate_doc_ids,
                    'cutoff': similarity_cutoff,
                    'limit': top_k
                }
            else:
                where_clause = "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, '')) @@ to_tsquery('english', :query)"
                query_params = {
                    'query': or_query,
                    'cutoff': similarity_cutoff,
                    'limit': top_k
                }
            
            search_results = session.execute(text(f"""
                WITH search_results AS (
                    SELECT
                        doc_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        published_date,
                        extra_metadata as metadata,
                        fts_rank(
                            title,
                            abstract,
                            to_tsquery('english', :query)
                        ) as score,
                        ts_headline(
                            'english',
                            coalesce(title, '') || ' ' || coalesce(abstract, ''),
                            to_tsquery('english', :query),
                            'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20'
                        ) as headline
                    FROM papers
                    WHERE {where_clause}
                )
                SELECT *
                FROM search_results
                WHERE score >= :cutoff
                ORDER BY score DESC
                LIMIT :limit
            """), query_params)
            
            # Process results
            results = []
            for row in search_results:
                result_dict = {
                    'doc_id': row.doc_id,
                    'score': float(row.score),
                    'metadata': {
                        'title': row.title,
                        'abstract': row.abstract,
                        'authors': row.authors,
                        'categories': row.categories,
                        'published_date': row.published_date,
                        **(row.metadata or {})
                    },
                    'matched_text': row.headline
                }
                results.append(result_dict)
                logger.debug(f"Found result: {result_dict['doc_id']} with score {result_dict['score']}")

            if candidate_doc_ids:
                logger.info(f"Search completed with filters: {len(results)} results found from {len(candidate_doc_ids)} candidate documents")
            else:
                logger.info(f"Search completed without filters: {len(results)} results found from all documents")
            return results

        except Exception as e:
            logger.error(f"Search with filter first failed: {str(e)}")
            return []
        finally:
            session.close()