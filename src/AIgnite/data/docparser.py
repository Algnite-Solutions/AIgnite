from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk
from uuid import uuid4
from pathlib import Path


class ArxivHTMLExtractor():
    #TODO: @rongcan, finish this extracter and serialize into a json
    def __init__(self):
        self.docs = []

    def load_html(self, source: str) -> str:
        with open(source, "r", encoding="utf-8") as f:
            return f.read()

    def extract_docset(self, html: str) -> DocSet:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("h1").text.strip()
        abstract = soup.find("blockquote", class_="abstract").text.strip().replace("Abstract:", "")

        text_chunks = [TextChunk(id=str(uuid4()), type="text", text=abstract)]

        self.doc.append(DocSet(
            arxiv_id="temp-id",
            title=title,
            authors=[],
            categories=[],
            published_date="2025-01-01",
            abstract=abstract,
            chunks=text_chunks,
        ))
    
    def serialize_docs(self, output_dir: str):
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.json(indent=4))