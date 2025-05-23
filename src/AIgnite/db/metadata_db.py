from typing import Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, JSON, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import os

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    doc_id = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    abstract = Column(Text)
    authors = Column(JSON)  # Store as JSON array
    categories = Column(JSON)  # Store as JSON array
    published_date = Column(String)
    pdf_data = Column(LargeBinary)  # Store PDF binary data
    chunk_ids = Column(JSON)  # Store all chunk IDs as JSON array
    image_ids = Column(JSON)  # Store all image IDs as JSON array

class MetadataDB:
    def __init__(self, db_path: str = None):
        """Initialize the metadata database.
        
        Args:
            db_path: PostgreSQL connection string. If None, uses default connection.
        """
        if db_path is None:
            # Use default connection string
            db_path = "postgresql://postgres:postgres@localhost:5432/aignite"
            
        self.engine = create_engine(db_path)
        self.Session = sessionmaker(bind=self.engine)
        self._init_db()
        
    def _init_db(self):
        """Initialize the database schema."""
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logging.error(f"Failed to initialize metadata database: {str(e)}")
            raise RuntimeError(f"Database initialization failed: {str(e)}")

    def save_paper(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any]) -> bool:
        """Save paper PDF and metadata.
        
        Args:
            doc_id: Document ID
            pdf_path: Path to PDF file
            metadata: Dictionary containing paper metadata
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            # Read PDF binary data
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()

            # Create or update paper
            paper = session.query(Paper).filter_by(doc_id=doc_id).first()
            if paper:
                # Update existing paper
                paper.title = metadata['title']
                paper.abstract = metadata['abstract']
                paper.authors = metadata['authors']
                paper.categories = metadata['categories']
                paper.published_date = metadata['published_date']
                paper.pdf_data = pdf_data
                paper.chunk_ids = metadata.get('chunk_ids', [])
                paper.image_ids = metadata.get('image_ids', [])
            else:
                # Create new paper
                paper = Paper(
                    doc_id=doc_id,
                    title=metadata['title'],
                    abstract=metadata['abstract'],
                    authors=metadata['authors'],
                    categories=metadata['categories'],
                    published_date=metadata['published_date'],
                    pdf_data=pdf_data,
                    chunk_ids=metadata.get('chunk_ids', []),
                    image_ids=metadata.get('image_ids', [])
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
            paper = session.query(Paper).filter_by(doc_id=doc_id).first()
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

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        session = self.Session()
        try:
            paper = session.query(Paper).filter_by(doc_id=doc_id).first()
            if not paper:
                return None
                
            return {
                'doc_id': paper.doc_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'categories': paper.categories,
                'published_date': paper.published_date,
                'chunk_ids': paper.chunk_ids,
                'image_ids': paper.image_ids
            }
        except Exception as e:
            logging.error(f"Failed to get metadata for doc_id {doc_id}: {str(e)}")
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
            paper = session.query(Paper).filter_by(doc_id=doc_id).first()
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