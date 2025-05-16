from typing import List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"


class BaseChunk(BaseModel):
    id: str
    type: ChunkType
    title: Optional[str] = None
    caption: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class TextChunk(BaseChunk):
    type: Literal[ChunkType.TEXT]
    text: str


class FigureChunk(BaseChunk):
    type: Literal[ChunkType.FIGURE]
    image_path: str  # e.g. local path, S3 url
    alt_text: Optional[str] = None


class TableChunk(BaseChunk):
    type: Literal[ChunkType.TABLE]
    table_html: Optional[str] = None # HTML representation of the table


Chunk = Union[TextChunk, FigureChunk, TableChunk]


class DocSet(BaseModel):
    doc_id: str
    title: str
    authors: List[str]
    categories: List[str]
    published_date: str
    abstract: str
    text_chunks: List[Chunk] = Field(default_factory=list)
    figure_chunks: List[Chunk] = Field(default_factory=list)
    table_chunks: List[Chunk] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    pdf_path: str

class DocSetList(BaseModel):
    docsets: List[DocSet]