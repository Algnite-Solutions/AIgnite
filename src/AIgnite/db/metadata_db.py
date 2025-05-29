"""Database modules for AIgnite."""
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, String, Integer, JSON, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declared_attr
import logging
import os
from ..data.docset import DocSet, BaseModel, Field

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
            HTML_path=docset.HTML_path
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
            'metadata': self.extra_metadata
        }

class MetadataDB:
    def __init__(self, db_path: str = None):
        """Initialize database connection.
        
        Args:
            db_path: Database connection string
        """
        if not db_path:
            db_path = "postgresql://postgres:11111@localhost:5432/aignite"
            
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_paper(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any] | DocSet) -> bool:
        """Save paper PDF and metadata.
        
        Args:
            doc_id: Document ID
            pdf_path: Path to PDF file
            metadata: Dictionary containing paper metadata or DocSet object
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            # Read PDF binary data
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()

            # Create or update paper
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            
            if isinstance(metadata, DocSet):
                if paper:
                    # Update existing paper
                    paper.title = metadata.title
                    paper.abstract = metadata.abstract
                    paper.authors = metadata.authors
                    paper.categories = metadata.categories
                    paper.published_date = metadata.published_date
                    paper.pdf_data = pdf_data
                    paper.chunk_ids = [chunk.id for chunk in metadata.text_chunks]
                    paper.figure_ids = [chunk.id for chunk in metadata.figure_chunks]
                    paper.table_ids = [chunk.id for chunk in metadata.table_chunks]
                    paper.extra_metadata = metadata.metadata
                    paper.pdf_path = metadata.pdf_path
                    paper.HTML_path = metadata.HTML_path
                else:
                    # Create new paper
                    paper = TableSchema.from_docset(metadata, pdf_data)
                    session.add(paper)
            else:
                # Handle dictionary metadata (for backward compatibility)
                if paper:
                    paper.title = metadata['title']
                    paper.abstract = metadata['abstract']
                    paper.authors = metadata['authors']
                    paper.categories = metadata['categories']
                    paper.published_date = metadata['published_date']
                    paper.pdf_data = pdf_data
                    paper.chunk_ids = metadata.get('chunk_ids', [])
                    paper.figure_ids = metadata.get('figure_ids', [])
                    paper.table_ids = metadata.get('table_ids', [])
                    paper.extra_metadata = metadata.get('metadata', {})
                else:
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
                        pdf_path=metadata.get('pdf_path', ''),
                        HTML_path=metadata.get('HTML_path')
                    )
                    session.add(paper)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to save paper {doc_id}: {str(e)}")
            return False
        finally:
            session.close()

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
        """Delete paper and its metadata.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            paper = session.query(TableSchema).filter_by(doc_id=doc_id).first()
            if not paper:
                return False
            
            session.delete(paper)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to delete paper {doc_id}: {str(e)}")
            return False
        finally:
            session.close() 