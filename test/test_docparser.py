import unittest
from pathlib import Path
from AIgnite.data.docparser import ArxivHTMLExtractor
from bs4 import BeautifulSoup
#new
from concurrent.futures import ProcessPoolExecutor, as_completed  
from pathlib import Path

class TestArxivHTMLExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ArxivHTMLExtractor()
        self.sample_html = self.extractor.load_html("/app/test/html_doc/test.txt")#Please change it to your path here and uncomment in the __main__

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

def process_single_html(local_file_path: Path, output_dir: Path):
    extractor = ArxivHTMLExtractor()
    html = extractor.load_html(local_file_path)
    extractor.extract_docset(html, output_dir)
    extractor.serialize_docs(output_dir)


def batch_process_htmls(input_dir: str, output_dir: str, max_workers: int = 8):  
    input_dir = Path(input_dir)  
    #output_dir = Path(output_dir)  
    html_files = list(input_dir.glob("*.txt"))  
    print(html_files)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  
        futures = [executor.submit(process_single_html, html_path, Path(output_dir)) for html_path in html_files]

        for future in as_completed(futures):  
            try:  
                future.result()  
            except Exception as e:  
                print(f"[ERROR] {e}")  

if __name__ == "__main__":
    #unittest.main()

    a = ArxivHTMLExtractor()#无意义，为了调用函数
    #首次启动的时候，uncomment下面的注释用来加载html
    #a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20376","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20201","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("https://ar5iv.labs.arxiv.org/html/2503.09516","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("https://ar5iv.labs.arxiv.org/html/2502.18017","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("your_test_url","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("your_test_url","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("your_test_url","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    #a.download_html("your_test_url","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")

    #下面二选一去uncomment，测试两个函数的效率
    #process_single_html("https://ar5iv.labs.arxiv.org/html/1907.01989","/data3/peirongcan/paperIgnite/AIgnite/test/htmls","/data3/peirongcan/paperIgnite/AIgnite/test/tem")
    #batch_process_htmls("/data3/peirongcan/paperIgnite/AIgnite/test/htmls","/data3/peirongcan/paperIgnite/AIgnite/test/tem")
