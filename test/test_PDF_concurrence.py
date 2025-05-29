import unittest
import os
from AIgnite.data.docparser_new import *
import json
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

def divide_one_day_into(date: str, count: int):
    time_sec = []
    time_last = 24 / count
    for i in range(count):
        clock = int(i * time_last)
        if clock >= 10:
            time_sec.append(date + str(clock) + "00")
        else:
            time_sec.append(date + "0" + str(clock) + "00")
    time_sec.append(date + "2359")
    return time_sec

class TestArxivHTMLExtractorParallel(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        self.html_text_folder = os.path.join(base_dir, "htmls")
        self.pdf_folder_path = os.path.join(base_dir, "pdfs")
        self.image_folder_path = os.path.join(base_dir, "imgs")
        self.json_output_path = os.path.join(base_dir, "jsons")
        self.arxiv_pool_path = os.path.join(base_dir, "html_url_storage/html_urls.txt")
        self.ak = 
        self.sk = 

        today = datetime.now(timezone.utc).date() - timedelta(days=2)
        self.date_str = today.strftime("%Y%m%d")
        self.time_slots = divide_one_day_into(self.date_str, 3)

        for path in [self.html_text_folder, self.pdf_folder_path, self.image_folder_path, self.json_output_path]:
            os.makedirs(path, exist_ok=True)

    def run_extractor_for_timeslot(self, start_str, end_str):
        extractor = ArxivPDFExtractor(None, self.pdf_folder_path, self.image_folder_path, self.arxiv_pool_path, self.json_output_path, self.ak, self.sk, start_str, end_str)
        extractor.extract_all()

    def test_parallel_extraction(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(len(self.time_slots) - 1):
                start_str = self.time_slots[i]
                end_str = self.time_slots[i + 1]
                futures.append(executor.submit(self.run_extractor_for_timeslot, start_str, end_str))

            for f in futures:
                f.result()

        # 检查输出 JSON 是否合理
        files = os.listdir(self.json_output_path)
        self.assertTrue(len(files) > 0, "No JSON output found. Check date range or arXiv response.")

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
