import unittest
from pathlib import Path
from AIgnite.data.docparser import ArxivHTMLExtractor
from bs4 import BeautifulSoup

class TestArxivHTMLExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ArxivHTMLExtractor()
        self.sample_html = self.extractor.load_html("/app/test/html_doc/test.txt")#Please change it to your path here

    def test_extract_docset(self):
        self.extractor.extract_docset(self.sample_html)

        self.assertEqual(len(self.extractor.docs), 1)

        doc = self.extractor.docs[0]
        
        real_abs = "Retrieval-augmented generation (RAG) has shown great potential for knowledge-intensive tasks, but its traditional architectures rely on static retrieval, limiting their effectiveness for complex questions that require sequential information-seeking. While agentic reasoning and search offer a more adaptive approach, most existing methods depend heavily on prompt engineering. In this work, we introduce RAG-Gym, a unified optimization framework that enhances information-seeking agents through fine-grained process supervision at each search step. We also propose ReSearch, a novel agent architecture that synergizes answer reasoning and search query generation within the RAG-Gym framework. Experiments on four challenging datasets show that RAG-Gym improves performance by up to 25.6% across various agent architectures, with ReSearch consistently outperforming existing baselines. Further analysis highlights the effectiveness of advanced LLMs as process reward judges and the transferability of trained reward models as verifiers for different LLMs. Additionally, we examine the scaling properties of training and inference in agentic RAG.\nThe project homepage is available at https://rag-gym.github.io/."

        self.assertEqual(doc.title, "RAG-Gym: Optimizing Reasoning and Search Agents with Process Supervision")
        self.assertEqual(doc.abstract, real_abs)
        self.assertEqual(doc.doc_id, "2502.13957")
        self.assertEqual(doc.authors,['Guangzhi Xiong', 'Qiao Jin', 'Xiao Wang', 'Yin Fang', 'Haolin Liu', 'Yifan Yang', 'Fangyuan Chen', 'Zhixing Song', 'Dengyu Wang', 'Minjia Zhang', 'Zhiyong Lu', 'Aidong Zhang'])
        self.assertEqual(doc.categories, ['cs.CL', 'cs.AI'])
        self.assertEqual(doc.published_date, "2025-02-19")

    def tearDown(self):
        self.extractor = None

if __name__ == "__main__":
    unittest.main()