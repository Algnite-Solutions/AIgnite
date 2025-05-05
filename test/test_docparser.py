import unittest
from pathlib import Path
from AIgnite.data.docparser import ArxivHTMLExtractor
from bs4 import BeautifulSoup
#new
from concurrent.futures import ProcessPoolExecutor, as_completed  
from pathlib import Path
import os

def test_process_single_html(local_file_path: Path, output_dir: Path, pdf_path: Path):
    """
    Function to process a single HTML file.
    Args:
        local_file_path (Path): The path of the local HTML file.
        output_dir (Path): The path of the output directory for saving the extracted documents and related files.
    Returns:
        DocSet: A DocSet object containing the information extracted from the HTML file.
    """
    extractor = ArxivHTMLExtractor()
    html = extractor.load_html(local_file_path)
    docs = extractor.extract_docset(html, output_dir, pdf_path)
    extractor.serialize_docs(output_dir)
    return docs

def test_batch_process_htmls(input_dir: str, output_dir: str, pdf_path: str, max_workers: int = 9):  
    """
    Function to batch process multiple HTML files using a process pool for parallel processing.
    Args:
        input_dir (str): The path of the input directory containing HTML files.
        output_dir (str): The path of the output directory for saving the extracted documents and related files.
        max_workers (int, optional): The maximum number of worker processes in the process pool, default is 9.
    Returns:
        None: This function does not return a value but processes and saves the extraction results of all HTML files.
    """
    input_dir = Path(input_dir)  
    #output_dir = Path(output_dir)  
    html_files = list(input_dir.glob("*.html"))  

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  
        futures = [executor.submit(test_process_single_html, html_path, Path(output_dir), pdf_path) for html_path in html_files]

        for future in as_completed(futures):  
            try:  
                future.result()  
            except Exception as e:  
                print(f"[ERROR] {e}")

class TestArxivHTMLExtractor(unittest.TestCase):

    def setUp(self):

        # The current directory where test_docparser.py is located
        base_dir = os.path.dirname(__file__) 
        self.extractor = ArxivHTMLExtractor()
        self.Your_htmls_folder_path = str(os.path.join(base_dir, "htmls"))
        self.Your_output_path = str(os.path.join(base_dir, "tem"))
        self.Your_pdf_folder_path = str(os.path.join(base_dir, "pdfs"))

    def test_correctness_single(self):
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2502.13957",self.Your_htmls_folder_path)
        docs = test_process_single_html(self.Your_htmls_folder_path+"/2502.13957.html",self.Your_output_path, self.Your_pdf_folder_path)
        self.assertEqual(docs.doc_id, "2502.13957")
        self.assertEqual(docs.title, "RAG-Gym: Optimizing Reasoning and Search Agents with Process Supervision")
        self.assertEqual(docs.authors, ["Guangzhi Xiong","Qiao Jin","Xiao Wang","Yin Fang","Haolin Liu","Yifan Yang","Fangyuan Chen","Zhixing Song","Dengyu Wang","Minjia Zhang","Zhiyong Lu","Aidong Zhang"])
        self.assertEqual(docs.categories,  ["cs.CL","cs.AI"])
        self.assertEqual(docs.published_date,  "2025-02-19")
        self.assertEqual(docs.abstract, "Retrieval-augmented generation (RAG) has shown great potential for knowledge-intensive tasks, but its traditional architectures rely on static retrieval, limiting their effectiveness for complex questions that require sequential information-seeking. While agentic reasoning and search offer a more adaptive approach, most existing methods depend heavily on prompt engineering. In this work, we introduce RAG-Gym, a unified optimization framework that enhances information-seeking agents through fine-grained process supervision at each search step. We also propose ReSearch, a novel agent architecture that synergizes answer reasoning and search query generation within the RAG-Gym framework. Experiments on four challenging datasets show that RAG-Gym improves performance by up to 25.6% across various agent architectures, with ReSearch consistently outperforming existing baselines. Further analysis highlights the effectiveness of advanced LLMs as process reward judges and the transferability of trained reward models as verifiers for different LLMs. Additionally, we examine the scaling properties of training and inference in agentic RAG.\nThe project homepage is available at https://rag-gym.github.io/.")

    def test_extract_docset_many(self):

        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2503.20376",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2503.20201",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2503.09516",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2501.11216",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2501.14733",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2501.15797",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2502.01113",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2502.03948",self.Your_htmls_folder_path)
        self.extractor.download_html("https://ar5iv.labs.arxiv.org/html/2502.13957",self.Your_htmls_folder_path)

        test_batch_process_htmls(self.Your_htmls_folder_path,self.Your_output_path,self.Your_pdf_folder_path)

    
    def tearDown(self):
        self.extractor = None
 

if __name__ == "__main__":
    unittest.main()
