from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,RootModel
from typing import List, Dict, Any, Optional
import uvicorn

from .paper_indexer import PaperIndexer
from ..data.docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType

# --- Initialize API and PaperIndexer ---
app = FastAPI()
indexer = PaperIndexer()

# --- Request Models ---
class ChunkInput(BaseModel):
    type: str
    text: str

class DocSetInput(BaseModel):
    doc_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: str
    pdf_path: str
    text_chunks: List[TextChunk]
    figure_chunks: List[FigureChunk]
    table_chunks: List[TableChunk]
    metadata: Optional[Dict[str, Any]] = {}

class DocSetList(RootModel):
    root: List[DocSetInput]

class SimilarQuery(BaseModel):
    query: str
    top_k: int = 5
    similarity_cutoff: float = 0.8

# --- API Endpoints ---
@app.post("/index_papers/")
async def index_papers(docset_list: DocSetList):
    try:
        papers = docset_list.root
        docsets = []
        for paper in papers:
            text_chunks = [TextChunk(**chunk.dict()) for chunk in paper.text_chunks]
            figure_chunks = [FigureChunk(**chunk.dict()) for chunk in paper.figure_chunks]
            table_chunks = [TableChunk(**chunk.dict()) for chunk in paper.table_chunks]

            docsets.append(DocSet(
                doc_id=paper.doc_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                categories=paper.categories,
                published_date=paper.published_date,
                pdf_path=paper.pdf_path,
                text_chunks=text_chunks,
                figure_chunks=figure_chunks,
                table_chunks=table_chunks,
                metadata=paper.metadata or {}
            ))
        indexer.index_papers(docsets)
        return {"message": f"{len(papers)} papers indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_metadata/{doc_id}")
async def get_paper_metadata(doc_id: str):
    metadata = indexer.get_paper_metadata(doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Paper metadata not found")
    return metadata

@app.post("/find_similar/")
async def find_similar(query: SimilarQuery):
    try:
        results = indexer.find_similar_papers(
            query=query.query,
            top_k=query.top_k,
            similarity_cutoff=query.similarity_cutoff
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))