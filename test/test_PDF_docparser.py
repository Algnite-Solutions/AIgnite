import unittest
import os
from AIgnite.data.docparser_new import *
import json

class TestArxivPDFExtractor(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        self.html_text_folder = os.path.join(base_dir, "htmls")
        self.pdf_folder_path = os.path.join(base_dir, "pdfs")
        self.image_folder_path = os.path.join(base_dir, "imgs")
        self.json_output_path = os.path.join(base_dir, "jsons")
        self.arxiv_pool_path = os.path.join(base_dir, "html_url_storage/html_urls.txt")
        self.ak = 
        self.sk = 

        today = datetime.now(timezone.utc).date()
        one_day = timedelta(days=2)
        today = today - one_day
        
        start_time = "0610"
        end_time = "0620"
        start_str = today.strftime("%Y%m%d") + start_time
        end_str = today.strftime("%Y%m%d") + end_time
        print(f"from {start_str} to {end_str}")

        for path in [self.html_text_folder, self.pdf_folder_path, self.image_folder_path, self.json_output_path]:
            os.makedirs(path, exist_ok=True)

        self.extractor = ArxivPDFExtractor(None, self.pdf_folder_path, self.image_folder_path, self.arxiv_pool_path, self.json_output_path, self.ak, self.sk, start_str, end_str)

    def test_end_to_end_extraction(self):

        self.extractor.extract_all()

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
