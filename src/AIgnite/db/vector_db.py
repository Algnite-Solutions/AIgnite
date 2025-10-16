# Cursor: follow LangChain FAISS rule (.cursor/rules/faiss_vectorstore.md)

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from FlagEmbedding import FlagModel
import logging
import os
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, AutoModel
from gritlm import GritLM

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

class GritLMEmbeddings(Embeddings):
    """GritLM embedding model wrapper for LangChain compatibility."""
    
    def __init__(self, model_name: str = 'GritLM/GritLM-7B', query_instruction: str = "Given a scientific paper title, retrieve the paper's abstract"):
        """Initialize GritLM embeddings.
        
        Args:
            model_name: Name of the GritLM model to use
            query_instruction: Instruction to use for query embeddings
        """
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize using GritLM official library
            self.model = GritLM(model_name, torch_dtype="auto")
            
            # Set use_cache to False for better performance
            try:
                self.model.model.config.use_cache = False
            except AttributeError:
                self.model.config.use_cache = False
                
            logger.info(f"Successfully loaded GritLM model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load GritLM model {model_name}: {str(e)}")
            raise
        
    def gritlm_instruction(self, instruction: str) -> str:
        """Format instruction for GritLM.
        
        Args:
            instruction: Instruction text
            
        Returns:
            Formatted instruction string
        """
        #print("ENTERING GRITLM INSTRUCTION MODE:")
        #print("INSTRUCTION:", instruction)
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Clean and normalize texts
            cleaned_texts = [text.strip() for text in texts]
            
            # Get embeddings using GritLM's encode method with empty instruction for documents
            embeddings = self.model.encode(cleaned_texts, instruction=self.gritlm_instruction(""))
            
            # Convert to numpy array for normalization
            embeddings = np.array(embeddings, dtype=np.float32)
            
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
            text = text.strip()
            
            # Get embedding using GritLM's encode method with query instruction
            embedding = self.model.encode([text], instruction=self.gritlm_instruction(self.query_instruction))[0]
            
            # Convert to numpy array for normalization
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
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
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
        if "bge" in model_name.lower():
            self.embeddings = BGEEmbeddings(model_name)
        elif "gritlm" in model_name.lower():
            self.embeddings = GritLMEmbeddings(model_name)
        else:
            self.embeddings = FlagModel(model_name)
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
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Only create directory if there's a directory path
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created directory for new vector database at {db_dir}")
            else:
                print(f"Using current directory for vector database: {self.db_path}")
        
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
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Only create directory if there's a directory path
                os.makedirs(db_dir, exist_ok=True)
            
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

    def get_all_doc_ids(self) -> List[str]:
        """Get all unique document IDs from the vector database.
        
        Returns:
            List of unique document IDs stored in the vector database
        """
        try:
            if not hasattr(self.faiss_store, 'docstore') or not hasattr(self.faiss_store.docstore, '_dict'):
                logger.warning("Vector database docstore not accessible")
                return []
            
            # Extract unique doc_ids from docstore
            doc_ids = set()
            for doc in self.faiss_store.docstore._dict.values():
                if hasattr(doc, 'metadata') and 'doc_id' in doc.metadata:
                    doc_ids.add(doc.metadata['doc_id'])
            
            doc_ids_list = sorted(list(doc_ids))
            logger.info(f"Retrieved {len(doc_ids_list)} unique document IDs from vector database")
            return doc_ids_list
            
        except Exception as e:
            logger.error(f"Failed to get all doc_ids from vector database: {str(e)}")
            return []

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
            # Check if we need to use FAISS native filtering
            if filters and ("include" in filters or "exclude" in filters):
                # FILTERED SEARCH PATH: Use FAISS IDSelector
                
                # Step 1: Collect target docstore_ids based on filter
                target_docstore_ids = set()
                
                if "include" in filters and "doc_ids" in filters["include"]:
                    # Include mode: collect docstore_ids matching the doc_ids
                    filter_doc_ids = set(filters["include"]["doc_ids"])
                    for docstore_id, doc in self.faiss_store.docstore._dict.items():
                        if hasattr(doc, 'metadata') and doc.metadata.get("doc_id") in filter_doc_ids:
                            target_docstore_ids.add(docstore_id)
                
                elif "exclude" in filters and "doc_ids" in filters["exclude"]:
                    # Exclude mode: collect all docstore_ids except those matching doc_ids
                    exclude_doc_ids = set(filters["exclude"]["doc_ids"])
                    for docstore_id, doc in self.faiss_store.docstore._dict.items():
                        if hasattr(doc, 'metadata') and doc.metadata.get("doc_id") not in exclude_doc_ids:
                            target_docstore_ids.add(docstore_id)
                #print("ENTER FITLERING MODE:")
                #print("TARGET DOCSTORE IDS:", target_docstore_ids)
                #print("NUM OF TARGET DOCSTORE IDS:", len(target_docstore_ids))
                # Step 2: Build reverse mapping from docstore_id to FAISS index
                docstore_id_to_faiss_idx = {
                    v: k for k, v in self.faiss_store.index_to_docstore_id.items()
                }
                
                # Step 3: Convert docstore_ids to FAISS numerical indices
                faiss_indices = []
                for ds_id in target_docstore_ids:
                    if ds_id in docstore_id_to_faiss_idx:
                        faiss_indices.append(docstore_id_to_faiss_idx[ds_id])
                
                # Step 4: Handle empty filter result
                if not faiss_indices:
                    logger.warning("Filter resulted in empty ID set, returning empty results")
                    return []
                
                # Step 5: Create IDSelector using FAISS built-in types
                # Convert to sorted numpy array for IDSelectorArray
                faiss_indices_array = np.array(sorted(faiss_indices), dtype=np.int64)
                
                if "include" in filters:
                    # Include mode: use IDSelectorArray directly
                    selector = faiss.IDSelectorArray(len(faiss_indices_array), faiss.swig_ptr(faiss_indices_array))
                else:  # exclude
                    # Exclude mode: wrap IDSelectorArray with NOT
                    base_selector = faiss.IDSelectorArray(len(faiss_indices_array), faiss.swig_ptr(faiss_indices_array))
                    selector = faiss.IDSelectorNot(base_selector)
                
                # Step 6: Get query embedding
                query_vector = self.embeddings.embed_query(query)
                query_vector = np.array([query_vector], dtype=np.float32)
                
                # Step 7: Create SearchParameters with selector
                search_params = faiss.SearchParameters()
                search_params.sel = selector
                
                # Step 8: Execute filtered search using search_c (low-level API)
                distances = np.zeros((query_vector.shape[0], top_k), dtype=np.float32)
                indices = np.zeros((query_vector.shape[0], top_k), dtype=np.int64)
                self.faiss_store.index.search_c(
                    query_vector.shape[0],
                    faiss.swig_ptr(query_vector),
                    top_k,
                    faiss.swig_ptr(distances),
                    faiss.swig_ptr(indices),
                    search_params
                )
                
                #print("NUM OF RESULTS:", len(indices[0]))
                #print("INDICES:", indices[0])
                # Step 9-13: Convert results to VectorEntry format
                results = []
                for i in range(len(indices[0])):
                    faiss_idx = indices[0][i]
                    distance = distances[0][i]
                    
                    # Skip invalid indices
                    if faiss_idx == -1:
                        continue
                    
                    # Get docstore_id
                    docstore_id = self.faiss_store.index_to_docstore_id.get(faiss_idx)
                    if not docstore_id:
                        continue
                    
                    # Get Document
                    doc = self.faiss_store.docstore._dict.get(docstore_id)
                    if not doc:
                        continue
                    
                    # Convert to VectorEntry
                    entry = self._document_to_vector_entry(doc)
                    
                    # For normalized vectors with inner product, distance is similarity
                    similarity_score = float(distance)
                    
                    results.append((entry, similarity_score))
                
                return results
                
            else:
                # NO FILTER PATH: Use LangChain wrapper directly
                docs_with_scores = self.faiss_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
                
                results = []
                for doc, score in docs_with_scores:
                    entry = self._document_to_vector_entry(doc)
                    results.append((entry, score))
                
                return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
