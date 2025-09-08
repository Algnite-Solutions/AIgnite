# Cursor: follow LangChain FAISS rule (.cursor/rules/faiss_vectorstore.md)

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from FlagEmbedding import FlagModel
import logging
import os
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class VectorEntry:
    """Class for storing vector database entries."""
    doc_id: str
    text: str
    text_type: str  # 'abstract' or 'chunk' or 'combined'
    chunk_id: Optional[str] = None
    vector: Optional[np.ndarray] = None

class BGEEmbeddings(Embeddings):
    """BGE embedding model wrapper for LangChain compatibility."""
    '''
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5',local_model_path: str = None):
        """Initialize BGE embeddings.
        
        Args:
            model_name: Name of the BGE model to use
        """
        self.model_name = model_name
        
        # Initialize embedding model
        #local_model_path = '~/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/'
        local_model_path = os.path.expanduser(local_model_path)
        if not os.path.exists(local_model_path):
            print(f"Downloading model to {local_model_path}")
            hf_hub_download(repo_id=model_name, local_dir=local_model_path)
        else:
            print(f"Found existing model at {local_model_path}")
        self.model = FlagModel(local_model_path)
    '''
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
        """Initialize BGE embeddings.
        
        Args:
            model_name: Name of the BGE model to use
        """
        self.model_name = model_name
        self.model = FlagModel(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Clean and normalize texts
            cleaned_texts = [text.lower().strip() for text in texts]
            
            # Get embeddings
            embeddings = self.model.encode(cleaned_texts)
            
            # Ensure proper shape and convert to float32
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            embeddings = embeddings.astype(np.float32)
            
            # Normalize the vectors
            faiss.normalize_L2(embeddings)
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Clean and normalize the text
            text = text.lower().strip()
            
            # Get embedding
            embedding = self.model.encode(text)
            
            # Ensure proper shape
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # Convert to float32
            embedding = embedding.astype(np.float32)
            
            # Normalize the vector
            faiss.normalize_L2(embedding)
            
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'model'):
            try:
                self.model.stop_self_pool()
            except:
                pass
            self.model = None

class VectorDB:
    """Vector database implementation using LangChain FAISS."""
    
    def __init__(self, db_path: str, model_name: str = 'BAAI/bge-base-en-v1.5', vector_dim: int = 768):
        """Initialize vector database with embedding model.
        
        Args:
            db_path: Path to save/load the vector database. Will try to load existing DB from this path first.
            model_name: Name of the embedding model to use
            vector_dim: Dimension of the embedding vectors (default: 768 for BGE base model)
            
        Raises:
            ValueError: If db_path is not provided
        """
        if not db_path:
            raise ValueError("db_path must be provided for VectorDB initialization")
            
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.model_name = model_name
        
        # Initialize embedding model
        self.embeddings = BGEEmbeddings(model_name)
        print(self.db_path)
        print(f"Initialized embeddings")
        # Try to load existing database first
        if self.exists():
            logger.info(f"Found existing vector database at {db_path}")
            if self.load():
                logger.info("Successfully loaded existing vector database")
                return
            else:
                logger.warning("Failed to load existing vector database, initializing new one")
        else:
            logger.info("No existing vector database found, initializing new one")
            # Create directory for new database if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            print(f"Created directory for new vector database at {self.db_path}")
        
        # Initialize new FAISS index if loading failed or database doesn't exist
        self.faiss_store = FAISS.from_texts(
            texts=["dummy"],  # Start with dummy text
            embedding=self.embeddings,
            metadatas=[{"doc_id": "dummy", "text_type": "dummy"}],
            ids=["dummy"]  # Specify custom ID for the dummy document
        )
        # Remove the dummy entry
        self.faiss_store.delete(["dummy"])
        
    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'embeddings'):
            self.embeddings = None
        if hasattr(self, 'faiss_store'):
            self.faiss_store = None

    def save(self) -> bool:
        """Save the vector database to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Save using LangChain FAISS save_local
            self.faiss_store.save_local(self.db_path, index_name="index")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save vector database: {str(e)}")
            return False

    def load(self) -> bool:
        """Load the vector database from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if files exist
            if not self.exists():
                logger.info("Vector database files don't exist")
                return False
                
            # Load using LangChain FAISS load_local
            self.faiss_store = FAISS.load_local(
                folder_path=self.db_path,
                embeddings=self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            return False

    def exists(self) -> bool:
        """Check if vector database files exist.
        
        Returns:
            bool: True if database files exist, False otherwise
        """
        index_file = os.path.join(self.db_path, "index.pkl")
        return os.path.exists(index_file)

    def _document_to_vector_entry(self, doc: Document) -> VectorEntry:
        """Convert LangChain Document to VectorEntry.
        
        Args:
            doc: LangChain Document object
            
        Returns:
            VectorEntry object
        """
        return VectorEntry(
            doc_id=doc.metadata.get("doc_id", ""),
            text=doc.page_content,
            text_type=doc.metadata.get("text_type", ""),
            chunk_id=doc.metadata.get("chunk_id")
        )

    def _vector_entry_to_document(self, entry: VectorEntry) -> Document:
        """Convert VectorEntry to LangChain Document.
        
        Args:
            entry: VectorEntry object
            
        Returns:
            LangChain Document object
        """
        metadata = {
            "doc_id": entry.doc_id,
            "text_type": entry.text_type
        }
        if entry.chunk_id:
            metadata["chunk_id"] = entry.chunk_id
            
        return Document(
            page_content=entry.text,
            metadata=metadata
        )
    # INPUT: doc_id,text_to_emb,metadata
    #   metadata: {doc_id:str, text_type:str, chunk_id:str}
    # OUTPUT: True if successful, False otherwise

    def add_document(self, vector_db_id: str, text_to_emb: str, doc_metadata: Dict[str, Any]) -> bool:
        """Add a document to the vector database.
        
        Args:
            vector_db_id: Vector database ID (unique identifier for the document)
            text_to_emb: Text content to embed and store
            metadata: Document metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document already exists
            '''
            existing_docs = self.faiss_store.similarity_search(
                query=vector_db_id,
                k=1,
                filter={"vector_db_id": vector_db_id}
            )
            if existing_docs:
                logger.warning(f"Document with vector_db_id {vector_db_id} already exists in vector database. Skip adding.")
                return False
            '''
            # Create Document object
            document = Document(
                page_content=text_to_emb,
                metadata=doc_metadata
            )
            
            # Add document to FAISS store
            self.faiss_store.add_documents([document], ids=[vector_db_id])
            
            logger.info(f"Successfully added document with vector_db_id: {vector_db_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document with vector_db_id {vector_db_id} to vector database: {str(e)}")
            return False
    '''
    def add_document(
        self,
        doc_id: str,
        abstract: str,
        text_chunks: List[str],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document and its chunks to the vector database.
        
        Args:
            doc_id: Document ID
            abstract: Document abstract
            text_chunks: List of text chunks
            metadata: Document metadata (not used for vectors)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document already exists
            existing_docs = self.faiss_store.similarity_search(
                query=doc_id,
                k=1,
                filter={"doc_id": doc_id}
            )
            if existing_docs:
                logger.warning(f"Document {doc_id} already exists in vector database. Skip adding.")
                return False
            
            # Create combined text for better document-level matching
            title = metadata.get('title', '')
            categories = ' '.join(metadata.get('categories', []))
            combined_text = f"{title} {categories} {abstract}"
            
            # Prepare documents to add
            documents_to_add = []
            
            # Add combined document
            combined_doc = Document(
                page_content=combined_text,
                metadata={
                    "doc_id": doc_id,
                    "text_type": "combined"
                }
            )
            documents_to_add.append(combined_doc)
            
            # Add abstract document
            abstract_doc = Document(
                page_content=abstract,
                metadata={
                    "doc_id": doc_id,
                    "text_type": "abstract"
                }
            )
            documents_to_add.append(abstract_doc)
            
            # Add text chunk documents
            chunk_ids = metadata.get('text_chunk_ids', [])
            for chunk_text, chunk_id in zip(text_chunks, chunk_ids):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "text_type": "chunk",
                        "chunk_id": chunk_id
                    }
                )
                documents_to_add.append(chunk_doc)
            
            # Add all documents to FAISS store
            self.faiss_store.add_documents(documents_to_add)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to vector database: {str(e)}")
            return False
    '''

    def collect_ids_to_delete(self, doc_id: str) -> List[str]:
        """Collect all IDs to delete for a document.
        
        Args:
            doc_id: Document ID
            
            Returns:
            List of IDs to delete
        """
        return [
            k for k, v in self.faiss_store.docstore._dict.items()
            if v.metadata.get("doc_id") == doc_id
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete all vectors for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all documents with this doc_id
            ids_to_delete = self.collect_ids_to_delete(doc_id)
            # 2. Delete those docs
            self.faiss_store.delete(ids=ids_to_delete)
            '''
            docs_to_delete = self.faiss_store.similarity_search(
                query="",  # Empty query to get all
                k=1000,  # Large number to get all results
                filter={"doc_id": doc_id}
            )
            
            if not docs_to_delete:
                return False
            
            # Get document IDs to delete
            doc_ids_to_delete = []
            for doc in docs_to_delete:
                # Generate unique ID for each document
                doc_id_to_delete = f"{doc.metadata['doc_id']}_{doc.metadata['text_type']}"
                if doc.metadata.get('chunk_id'):
                    doc_id_to_delete += f"_{doc.metadata['chunk_id']}"
                doc_ids_to_delete.append(doc_id_to_delete)
            
            # Delete documents
            self.faiss_store.delete(doc_ids_to_delete)
            '''
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id} from vector database: {str(e)}")
            return False

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Tuple[VectorEntry, float]]:
        """Search for similar vectors with filter-first approach.
        
        Args:
            query: Search query text
            filters: Optional filters to apply first (supports doc_ids, etc.) 
                    e.g., filters={"include": {"doc_ids": ["001", "004"]}} for inclusion
                    e.g., filters={"exclude": {"doc_ids": ["001", "004"]}} for exclusion
            top_k: Number of results to return
            
        Returns:
            List of tuples containing (VectorEntry, similarity_score)
        """
        try:
            # Prepare filter for LangChain FAISS
            faiss_filter = None
            if filters:
                # Convert filters to FAISS format
                if "include" in filters and "doc_ids" in filters["include"]:
                    faiss_filter = {"doc_id": filters["include"]["doc_ids"]}
                elif "exclude" in filters and "doc_ids" in filters["exclude"]:
                    faiss_filter = {"doc_id": {"$nin": filters["exclude"]["doc_ids"]}}
                else:
                    # FAISS doesn't support list filters directly, we'll handle this in post-processing
                    pass
                print(f"faiss_filter: {faiss_filter}")
            # Perform similarity search with scores
            docs_with_scores = self.faiss_store.similarity_search_with_score(
                query=query,
                k=top_k * 5,  # Get more results to account for filtering
                filter=faiss_filter
            )
            print('result before filters:', len(docs_with_scores))
            # Convert to VectorEntry format and apply additional filters
            results = []
            for doc, score in docs_with_scores:
                entry = self._document_to_vector_entry(doc)
                
                results.append((entry, float(score)))
            
            print('result after filters:', len(results))
            
            # Sort by similarity score (lower is better for distance)
            results.sort(key=lambda x: x[1])
            
            # Return top k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
