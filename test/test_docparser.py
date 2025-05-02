import unittest
from pathlib import Path
from AIgnite.data.docparser import ArxivHTMLExtractor
from bs4 import BeautifulSoup
#new
from concurrent.futures import ProcessPoolExecutor, as_completed  
from pathlib import Path

def test_process_single_html(local_file_path: Path, output_dir: Path):
    extractor = ArxivHTMLExtractor()
    html = extractor.load_html(local_file_path)
    extractor.extract_docset(html, output_dir)
    extractor.serialize_docs(output_dir)

def test_batch_process_htmls(input_dir: str, output_dir: str, max_workers: int = 8):  
    input_dir = Path(input_dir)  
    #output_dir = Path(output_dir)  
    html_files = list(input_dir.glob("*.html"))  

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  
        futures = [executor.submit(test_process_single_html, html_path, Path(output_dir)) for html_path in html_files]

        for future in as_completed(futures):  
            try:  
                future.result()  
            except Exception as e:  
                print(f"[ERROR] {e}")

class TestArxivHTMLExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ArxivHTMLExtractor()
        #Please change it to your path here and uncomment in the __main__
        self.sample_html = self.extractor.load_html("/data3/peirongcan/paperIgnite/AIgnite/test/htmls/2502.13957.html")

    def test_extract_docset(self):
         #无意义，为了调用函数
        a = ArxivHTMLExtractor()

        a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20376","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20201","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2503.09516","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2501.11216","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2501.14733","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2501.15797","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2502.01113","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2502.03948","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        a.download_html("https://ar5iv.labs.arxiv.org/html/2502.13957","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        

        #测试两个函数的效率
        test_process_single_html("/data3/peirongcan/paperIgnite/AIgnite/test/htmls/2502.13957.html","/data3/peirongcan/paperIgnite/AIgnite/test/tem")
        test_batch_process_htmls("/data3/peirongcan/paperIgnite/AIgnite/test/htmls","/data3/peirongcan/paperIgnite/AIgnite/test/tem")





    
    def tearDown(self):
        self.extractor = None
 

if __name__ == "__main__":
    unittest.main()
    '''a = ArxivHTMLExtractor()

    a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20376","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2503.20201","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2503.09516","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2501.11216","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2501.14733","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2501.15797","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2502.01113","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2502.03948","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
    a.download_html("https://ar5iv.labs.arxiv.org/html/2502.13957","/data3/peirongcan/paperIgnite/AIgnite/test/htmls")
        

    #测试两个函数的效率
    #test_process_single_html("/data3/peirongcan/paperIgnite/AIgnite/test/htmls/2502.13957.html","/data3/peirongcan/paperIgnite/AIgnite/test/tem")
    test_batch_process_htmls("/data3/peirongcan/paperIgnite/AIgnite/test/htmls","/data3/peirongcan/paperIgnite/AIgnite/test/tem")'''
