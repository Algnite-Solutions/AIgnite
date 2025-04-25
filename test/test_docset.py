import unittest
from AIgnite.data import DocSet
from AIgnite.data import TextChunk, FigureChunk, TableChunk, ChunkType


class TestDocSetModel(unittest.TestCase):

    def test_text_chunk_creation(self):
        chunk = TextChunk(id="1", type="text", text="Some text.")
        self.assertEqual(chunk.type, ChunkType.TEXT)
        self.assertEqual(chunk.text, "Some text.")
        self.assertIsInstance(chunk, TextChunk)

    def test_figure_chunk_creation(self):
        chunk = FigureChunk(id="fig1", type="figure", image_path="fig1.png", caption="A figure")
        self.assertEqual(chunk.type, ChunkType.FIGURE)
        self.assertEqual(chunk.image_path, "fig1.png")
        self.assertEqual(chunk.caption, "A figure")
        self.assertIsInstance(chunk, FigureChunk)

    def test_table_chunk_creation(self):
        chunk = TableChunk(
            id="table1",
            type="table",
            caption="Table 1",
            table_html="<table><tr><td>1</td></tr></table>",
        )
        self.assertEqual(chunk.type, ChunkType.TABLE)
        self.assertEqual(chunk.table_html, "<table><tr><td>1</td></tr></table>")
        self.assertIsInstance(chunk, TableChunk)

    def test_docset_construction(self):
        doc = DocSet(
            doc_id="1234.56789",
            title="Test Paper",
            authors=["Alice", "Bob"],
            categories=["cs.CL"],
            published_date="2025-01-01",
            abstract="This is an abstract.",
            chunks=[
                TextChunk(id="t1", type="text", text="Intro..."),
                FigureChunk(id="f1", type="figure", image_path="fig1.png", caption="A figure"),
            ]
        )
        self.assertEqual(doc.doc_id, "1234.56789")
        self.assertEqual(doc.title, "Test Paper")
        self.assertEqual(len(doc.chunks), 2)
        self.assertIsInstance(doc.chunks[0], TextChunk)
        self.assertIsInstance(doc.chunks[1], FigureChunk)


if __name__ == '__main__':
    unittest.main()