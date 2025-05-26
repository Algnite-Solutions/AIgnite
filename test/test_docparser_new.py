import unittest
import os
from pathlib import Path
from AIgnite.data.docparser_new import *
from bs4 import BeautifulSoup
import json

class TestArxivHTMLExtractor(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        self.html_text_folder = os.path.join(base_dir, "htmls")
        self.pdf_folder_path = os.path.join(base_dir, "pdfs")
        self.image_folder_path = os.path.join(base_dir, "imgs")
        self.json_output_path = os.path.join(base_dir, "jsons")
        self.arxiv_pool_path = os.path.join(base_dir, "html_url_storage/html_urls.txt")

        for path in [self.html_text_folder, self.pdf_folder_path, self.image_folder_path, self.json_output_path]:
            os.makedirs(path, exist_ok=True)

        # Initialize the pool file
        '''with open(self.arxiv_pool_path, 'w', encoding='utf-8') as f:
            f.write("") '''

        self.extractor = ArxivHTMLExtractor(
            html_text_folder=self.html_text_folder,
            pdf_folder_path=self.pdf_folder_path,
            arxiv_pool=self.arxiv_pool_path,
            image_folder_path=self.image_folder_path,
            json_path=self.json_output_path
        )
        self.extractor2 = ArxivPDFExtractor(None, self.pdf_folder_path, self.image_folder_path, self.arxiv_pool_path, self.json_output_path)

    def test_end_to_end_extraction(self):

        #choose which way to extract here.
        self.extractor.extract_all_htmls()
        #self.extractor2.extract_all()

        files = os.listdir(self.json_output_path)
        self.assertTrue(len(files) > 0, "No JSON output found. Maybe: 1. you can't search anything from the arxiv api. Please change the date you search. 2.All papers are already in the html_urls.txt")

        for file in files:
            with open(os.path.join(self.json_output_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertIn("doc_id", data)
                self.assertIn("title", data)
                self.assertTrue(data.get("title", "").strip() != "")
                self.assertIn("figure_chunks", data)
                self.assertIn("text_chunks", data)

class TestArxivPDFExtractor(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        self.html_text_folder = os.path.join(base_dir, "htmls")
        self.pdf_folder_path = os.path.join(base_dir, "pdfs")
        self.image_folder_path = os.path.join(base_dir, "imgs")
        self.json_output_path = os.path.join(base_dir, "jsons")
        self.arxiv_pool_path = os.path.join(base_dir, "html_url_storage/html_urls.txt")

        for path in [self.html_text_folder, self.pdf_folder_path, self.image_folder_path, self.json_output_path]:
            os.makedirs(path, exist_ok=True)

        # Initialize the pool file
        with open(self.arxiv_pool_path, 'w', encoding='utf-8') as f:
            f.write("") 

        self.extractor = ArxivHTMLExtractor(
            html_text_folder=self.html_text_folder,
            pdf_folder_path=self.pdf_folder_path,
            arxiv_pool=self.arxiv_pool_path,
            image_folder_path=self.image_folder_path,
            json_path=self.json_output_path
        )
        self.extractor2 = ArxivPDFExtractor(None, self.pdf_folder_path, self.image_folder_path, self.arxiv_pool_path, self.json_output_path)

    def test_end_to_end_extraction(self):

        #choose which way to extract here.
        #self.extractor.extract_all_htmls()
        self.extractor2.extract_all()

        files = os.listdir(self.json_output_path)
        self.assertTrue(len(files) > 0, "No JSON output found. Maybe: 1. you can't search anything from the arxiv api. Please change the date you search. 2.All papers are already in the html_urls.txt")

        for file in files:
            with open(os.path.join(self.json_output_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertIn("doc_id", data)
                self.assertIn("title", data)
                self.assertTrue(data.get("title", "").strip() != "")
                self.assertIn("figure_chunks", data)
                self.assertIn("text_chunks", data)

if __name__ == "__main__":
    unittest.main()
