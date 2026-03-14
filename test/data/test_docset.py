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
            pdf_path="a/b/c.pdf",
            HTML_path="d/e/f.html",
            text_chunks=[
                TextChunk(id="t1", type="text", text="Intro..."),
            ],
            figure_chunks=[
                FigureChunk(id="f1", type="figure", image_path="fig1.png", caption="A figure"),
            ],
            table_chunks=[]
        )
        self.assertEqual(doc.doc_id, "1234.56789")
        self.assertEqual(doc.title, "Test Paper")
        self.assertEqual(len(doc.text_chunks), 1)
        self.assertEqual(len(doc.figure_chunks), 1)
        self.assertEqual(len(doc.table_chunks), 0)
        self.assertIsInstance(doc.text_chunks[0], TextChunk)
        self.assertIsInstance(doc.figure_chunks[0], FigureChunk)

    def test_docset_with_multiple_chunks(self):
        """测试包含多个不同类型chunks的DocSet"""
        doc = DocSet(
            doc_id="2345.67890",
            title="Complex Paper",
            authors=["Charlie", "David", "Eve"],
            categories=["cs.AI", "cs.CL"],
            published_date="2025-02-01",
            abstract="A complex paper with multiple sections.",
            pdf_path="x/y/z.pdf",
            HTML_path="u/v/w.html",
            text_chunks=[
                TextChunk(id="t1", type="text", text="Introduction..."),
                TextChunk(id="t2", type="text", text="Methods..."),
                TextChunk(id="t3", type="text", text="Results...")
            ],
            figure_chunks=[
                FigureChunk(id="f1", type="figure", image_path="fig1.png", caption="Figure 1"),
                FigureChunk(id="f2", type="figure", image_path="fig2.png", caption="Figure 2")
            ],
            table_chunks=[
                TableChunk(id="t1", type="table", caption="Table 1", table_html="<table><tr><td>Data</td></tr></table>"),
                TableChunk(id="t2", type="table", caption="Table 2", table_html="<table><tr><td>Results</td></tr></table>")
            ]
        )
        
        # 验证基本信息
        self.assertEqual(doc.doc_id, "2345.67890")
        self.assertEqual(len(doc.authors), 3)
        self.assertEqual(len(doc.categories), 2)
        
        # 验证chunks数量
        self.assertEqual(len(doc.text_chunks), 3)
        self.assertEqual(len(doc.figure_chunks), 2)
        self.assertEqual(len(doc.table_chunks), 2)
        
        # 验证chunks内容
        self.assertEqual(doc.text_chunks[0].text, "Introduction...")
        self.assertEqual(doc.figure_chunks[0].caption, "Figure 1")
        self.assertEqual(doc.table_chunks[0].caption, "Table 1")

    def test_docset_with_empty_chunks(self):
        """测试所有chunks都为空的情况"""
        doc = DocSet(
            doc_id="3456.78901",
            title="Empty Paper",
            authors=["Frank"],
            categories=["cs.SE"],
            published_date="2025-03-01",
            abstract="A paper with no chunks.",
            pdf_path="p/q/r.pdf",
            HTML_path="s/t/u.html",
            text_chunks=[],
            figure_chunks=[],
            table_chunks=[]
        )
        
        # 验证chunks为空
        self.assertEqual(len(doc.text_chunks), 0)
        self.assertEqual(len(doc.figure_chunks), 0)
        self.assertEqual(len(doc.table_chunks), 0)
        
        # 验证其他属性
        self.assertEqual(doc.title, "Empty Paper")
        self.assertEqual(len(doc.authors), 1)
        self.assertEqual(doc.categories, ["cs.SE"])

    def test_docset_chunk_types(self):
        """测试不同类型chunks的属性"""
        doc = DocSet(
            doc_id="4567.89012",
            title="Type Test Paper",
            authors=["Grace"],
            categories=["cs.CV"],
            published_date="2025-04-01",
            abstract="Testing chunk types.",
            pdf_path="m/n/o.pdf",
            HTML_path="h/i/j.html",
            text_chunks=[
                TextChunk(id="t1", type="text", text="Test text", title="Section 1", caption="First section")
            ],
            figure_chunks=[
                FigureChunk(id="f1", type="figure", image_path="test.png", caption="Test figure", alt_text="Alternative text")
            ],
            table_chunks=[
                TableChunk(id="t1", type="table", caption="Test table", table_html="<table><tr><td>Test</td></tr></table>")
            ]
        )
        
        # 验证TextChunk属性
        text_chunk = doc.text_chunks[0]
        self.assertEqual(text_chunk.type, ChunkType.TEXT)
        self.assertEqual(text_chunk.title, "Section 1")
        self.assertEqual(text_chunk.caption, "First section")
        
        # 验证FigureChunk属性
        figure_chunk = doc.figure_chunks[0]
        self.assertEqual(figure_chunk.type, ChunkType.FIGURE)
        self.assertEqual(figure_chunk.alt_text, "Alternative text")
        
        # 验证TableChunk属性
        table_chunk = doc.table_chunks[0]
        self.assertEqual(table_chunk.type, ChunkType.TABLE)
        self.assertIn("Test", table_chunk.table_html)


if __name__ == '__main__':
    unittest.main()