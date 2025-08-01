"""Database modules for AIgnite."""
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, String, Integer, JSON, Text, LargeBinary
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
            blog=getattr(docset, 'blog', None)  # Support blog field if present
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
            'pdf_path': self.pdf_path,  # Add pdf_path
            'HTML_path': self.HTML_path  # Add HTML_path
        }

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

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        similarity_cutoff: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Search documents using PostgreSQL full-text search."""
        session = self.Session()
        try:
            # First, let's debug the query parsing
            debug_query = session.execute(text("""
                SELECT plainto_tsquery('english', :query) as parsed_query
            """), {'query': query}).scalar()
            logger.debug(f"Parsed query: {debug_query}")
            # Use OR (|) between words for more forgiving matching
            or_query = ' | '.join(query.split())

            # Modified search query with to_tsquery for OR logic
            search_results = session.execute(text("""
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
                    WHERE to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, '')) @@ 
                          to_tsquery('english', :query)
                )
                SELECT *
                FROM search_results
                WHERE score >= :cutoff
                ORDER BY score DESC
                LIMIT :limit
            """), {
                'query': or_query,  # Use OR between words
                'cutoff': similarity_cutoff,
                'limit': top_k
            })
            # Debug the results
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

            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
        finally:
            session.close()

    def add_document(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any]) -> bool:
        """Add a document PDF and metadata.
        Args:
            doc_id: Document ID
            pdf_path: Path to PDF file
            metadata: Dictionary containing document metadata with required fields:
                     title, abstract, authors, categories, published_date
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
            # Check if document already exists
            if session.query(TableSchema).filter_by(doc_id=doc_id).first():
                logging.warning(f"Document {doc_id} already exists. Skip saving.")
                return False

            # Read PDF binary data
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()

            # Create new document
            document = TableSchema(
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
                blog=getattr(metadata, 'blog', None)  # Support blog field if present
            )
            session.add(document)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to save document {doc_id}: {str(e)}")
            return False
        finally:
            session.close()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document.
        Args:
            doc_id: Document ID
        Returns:
            Dictionary containing metadata or None if not found
        """
        session = self.Session()
        try:
            document = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not document:
                return None
            return document.to_dict()
        except Exception as e:
            logging.error(f"Failed to get metadata for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close()

    def get_document_pdf(self, doc_id: str, save_path: str = None) -> Optional[bytes]:
        """Retrieve PDF data for a document.
        Args:
            doc_id: Document ID
            save_path: Optional path to save the PDF file
        Returns:
            PDF binary data if save_path is None, else None
        """
        session = self.Session()
        try:
            document = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not document or not document.pdf_data:
                return None

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(document.pdf_data)
                return None
            return document.pdf_data
        except Exception as e:
            logging.error(f"Failed to get PDF for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close()

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and its metadata.
        Args:
            doc_id: Document ID
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            document = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not document:
                return False
            
            session.delete(document)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to delete document {doc_id}: {str(e)}")
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
            document = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not document:
                return False
            document.blog = blog
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
            document = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not document:
                return None
            return document.blog
        except Exception as e:
            logging.error(f"Failed to get blog for doc_id {doc_id}: {str(e)}")
            return None
        finally:
            session.close() 