from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from .pdfparser import *
from pathlib import Path
#new
from datetime import datetime, timezone, timedelta
import arxiv
import os
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import re
from typing import List, Tuple
import base64
from volcengine.visual.VisualService import VisualService
from spire.pdf.common import *
from spire.pdf import *
from abc import ABC, abstractmethod

class BaseHTMLExtractor(ABC):
    """Abstract base classes of the HTML extractor"""
    def __init__(self, html_text_folder, pdf_folder_path, arxiv_pool, image_folder_path, json_path, volcengine_ak, volcengine_sk, start_time, end_time, max_results):
        '''
        Args:
        html_text_folder: the folder path used to store the .html file.
        pdf_folder_path: the folder path used to store the .pdf (and .md) file.
        arxiv_pool: path of a .txt file to store the arxiv_id which is serialized successfully
        image_folder_path:  the folder path used to store the .png file.
        json_path: the folder path used to store the .json file.
        volcengine_ak: get from https://console.volcengine.com/ai/ability/info/72
        volcengine_sk: get from https://console.volcengine.com/ai/ability/info/72
        start_time: the earliest paper you want
        end_time: the last paper you want
        '''
        self.date = datetime.now(timezone.utc).date()
        self.docs = []
        self.html_text_folder = html_text_folder
        self.pdf_folder_path = pdf_folder_path
        self.arxiv_pool = arxiv_pool
        self.image_folder_path = image_folder_path
        self.json_path = json_path
        self.start_time = start_time
        self.end_time = end_time
        #Helper
        self.pdf_parser_helper = ArxivPDFExtractor(self.docs, pdf_folder_path, image_folder_path, arxiv_pool, json_path, volcengine_ak, volcengine_sk, start_time, end_time)
        self.max_results = max_results

    @abstractmethod
    def extract_all_htmls(self) -> DocSet:
        """Carry out the complete extraction process"""
        pass

    @abstractmethod
    def init_docset(self):
        """Initialize the document metadata"""
        pass

    @abstractmethod
    def serialize_docs(self):
        """Serialize the extraction results into JSON"""
        pass

    @abstractmethod
    def extract_text(self, soup):
        """Extract text chunks from HTML"""
        pass

    @abstractmethod
    def extract_figures_to_folder(self, soup, img_path, arxivid):
        """Extract images from HTML and save them to a folder"""
        pass

    @abstractmethod
    def extract_tables(self, soup, arxivid):
        """Extract the table from HTML"""
        pass

class ArxivHTMLExtractor(BaseHTMLExtractor):
    """
    A class used to extract information from daily arXiv HTMLs and serialize it into JSON files.
    if there not exist a html then use pdf to extract the remain.
    """
    def init_docset(self):
        """
        Initialize the docset with papers' some metadata:
        doc_id, title, authors, categories, published_date, abstract, pdf_path, HTML_path
        The 3 types of chunk remain to add
        """
        client = arxiv.Client()

        query = "cat:cs.* AND submittedDate:[" + self.start_time + " TO " + self.end_time + "]"

        search = arxiv.Search(
            query=query,
            max_results=self.max_results,  # You can set max papers you want here
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        #print('only 3 papers')
        print(f"grabbing arXiv papers in cs.* submitted from {self.start_time} to {self.end_time}......")

        # Test if we have extracted already or not. Download pdf and try to download html
        tem = client.results(search)
        tem = list(tem)
        print("successful search!")

        for result in tem:
            html_url = result.pdf_url.replace("pdf", "html")
            arxiv_id = html_url.split('/')[-1]
            with open(self.arxiv_pool, "r", encoding="utf-8") as f:
                if arxiv_id in f.read():
                    print(f"{arxiv_id} is already extracted before!")
                    continue
            try:
                #add basic info
                add_doc = DocSet(
                doc_id=arxiv_id,
                title=result.title,
                authors=[author.name for author in result.authors],
                categories=result.categories,
                published_date=str(result.published),
                abstract=result.summary,
                pdf_path=str(os.path.join(self.pdf_folder_path, f'{arxiv_id}.pdf')),
                comments=result.comment,
                #Set htmlpath to None first and update it later
                HTML_path=None 
            )

                success = download_paper(
                    result=result,
                    save_path=self.pdf_folder_path,
                    filename=f"{arxiv_id}.pdf"
                )
                if not success:
                    add_doc.pdf_path = None
                    print(f"❌ 论文 {result.title} 下载最终失败")

                response = requests.get(html_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                #print(response.text)
                tag = soup.find('article')

                if tag is not None:
                    file_path = os.path.join(self.html_text_folder, f"{arxiv_id}.html")
                    with open(file_path, 'w', encoding='utf-8') as html_file:
                        html_file.write(str(tag))
                    add_doc.HTML_path = file_path
                    print(f"The HTML of {arxiv_id} has been saved to: {file_path}")
                else:
                    print(f"can not get {arxiv_id}'s html")
                    add_doc.HTML_path = None
                

                self.docs.append(add_doc)
            except Exception as e:
                self.docs.append(add_doc)
                print(f"request failed: {e}, DocSet will not include this HTML.")

    def extract_text(self, soup: BeautifulSoup):
        try:
            article = soup.find('article')
            all_text = []
            sections = article.find_all('section', class_ = ['ltx_section', 'ltx_appendix'])
            if sections:
                for section in sections:
                    # Remove the "figure" tag and its contents
                    for figure in section.find_all('figure'):
                        figure.extract()
                    section_text = section.get_text()
                    section_text = section_text.replace('\n\n', '\n')
                    # get id of section
                    section_id = section.get('id', '')  
                    title_elem = section.find('h2', class_='ltx_title ltx_title_section')
                    # get h2section's title
                    title = title_elem.get_text(strip=True) if title_elem else ''  

                    # based on the html structure of ar5iv, there is no obvious content that can be used as a caption.
                    caption = title 
                    all_text.append(TextChunk(
                        id=section_id,
                        type=ChunkType.TEXT,
                        title=title,
                        caption=caption,
                        text=section_text,
                    ))
                return all_text
            return None
        except Exception as e:
            print(f"Error when extracting text: {e}")
            return None

    def extract_figures_to_folder(self, soup, img_path, arxivid):
        figures = []

        for fig in soup.find_all(lambda tag: tag.name == 'figure' and 'ltx_table' not in tag.get('class', [])):
            img = fig.find('img')
            caption = fig.find('figcaption')
            fig_id = fig.get('id', '')

            if img and caption:
                tag = caption.find('span', class_='ltx_tag_figure')
                if tag and fig_id:
                    numbers = re.findall(r'\d+', fig_id)
                    if len(numbers) == 2:
                        figure_name = str(arxivid)+'_'+"Figure" + numbers[1]
                    elif len(numbers) > 2:
                        figure_name = str(arxivid)+'_'+"Figure" + numbers[1] + f'({numbers[2]})'
                    else:
                        figure_name = str(arxivid)+'_'+"Figure"

                    img_src = img['src']
                    #Get the complete image URL
                    img_url = urljoin(f"https://arxiv.org/html/{str(arxivid)}/", img_src) 
                    alt = img.get('alt', '')
                    caption_text = caption.get_text(strip=True)
                    #img_data = requests.get(img_url).content
                    img_data = get_img_from_url(arxivid,img_src)
                    #The file name of the stored picture
                    img_filename = os.path.join(img_path, f'{figure_name}.png')

                    #Make sure the image storage directory exists
                    os.makedirs(os.path.dirname(img_filename), exist_ok=True)

                    #Save the picture to the local machine
                    with open(img_filename, 'wb') as f:
                        if img_data:
                            f.write(img_data)
                
                    figures.append(FigureChunk(
                        id = fig_id,
                        title = figure_name,
                        type = ChunkType.FIGURE,
                        image_path = img_filename,
                        alt_text = alt,
                        caption = caption_text
                    ))

        return figures
    
    def extract_tables(self, soup, arxivid):
        tables = []
        for table_fig in soup.find_all('figure', class_='ltx_table'):
            table = table_fig.find('table')
            caption = table_fig.find('figcaption')
            table_id = table_fig.get('id', '')

            if table and caption:
                tag = caption.find('span', class_='ltx_tag_table')
                if tag:
                    table_name = tag.text.strip().rstrip(':').strip()
                    table_name = table_name.replace(' ', '') 
                    table_name = str(arxivid) + '_' + table_name
                    table_html = str(table)
                    caption_text = caption.get_text(strip=True)

                    tables.append(TableChunk(
                        id = table_id,
                        title = table_name,
                        type = ChunkType.TABLE,
                        table_html = table_html,
                        caption = caption_text
                    ))
        return tables
    
    def extract_all_htmls(self) -> DocSet:
        """
        All in one function.
        """
        self.init_docset()

        print("Init over. Now begin chunking...")

            # 读取已处理的论文列表，获取最后一行作为基准
        last_processed_id = None
        try:
            with open(self.arxiv_pool, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_processed_id = lines[-1].strip()
        except FileNotFoundError:
            pass

        # 提取基准ID的前9位数字
        baseline_number = None
        if last_processed_id:
            # 提取前9位数字，例如从 "2510.01878v1" 提取 "251001878"
            match = re.match(r'(\d{4})\.(\d{5})', last_processed_id)
            if match:
                baseline_number = int(match.group(1) + match.group(2))

        for filename in os.listdir(self.html_text_folder):
            #print("test2")
            if filename.endswith(".html"):
                arxiv_id = filename[:-5]
                
                # 如果有基准ID，只处理前9位数字更大的论文
                if baseline_number is not None:
                    match = re.match(r'(\d{4})\.(\d{5})', arxiv_id)
                    if match:
                        current_number = int(match.group(1) + match.group(2))

                        if current_number <= baseline_number:
                            print("continue")
                            continue
                file_path = os.path.join(self.html_text_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    soup = BeautifulSoup(html_content, "html.parser")

                    for docset in self.docs:
                        if docset.doc_id == filename[:-5] and docset.HTML_path is not None:
                            print(f"Processing {docset.doc_id}")
                            figurechunks = self.extract_figures_to_folder(soup,self.image_folder_path,docset.doc_id)
                            table_chunks = self.extract_tables(soup,docset.doc_id)
                            docset.figure_chunks = figurechunks
                            docset.table_chunks = table_chunks
                            docset.text_chunks = self.extract_text(soup)
                            
                            # 调试信息：检查每个字段是否为空并打印前50个字符
                            print(f"=== 调试信息 - {docset.doc_id} ===")
                            
                            # 检查 figure_chunks
                            if docset.figure_chunks is not None and len(docset.figure_chunks) > 0:
                                print(f"✅ figure_chunks: 有 {len(docset.figure_chunks)} 个元素")
                                for i, chunk in enumerate(docset.figure_chunks[:3]):  # 只显示前3个
                                    print(f"  Figure {i+1}: {str(chunk)[:50]}...")
                            else:
                                print("❌ figure_chunks: 为空或None")
                            
                            # 检查 table_chunks
                            if docset.table_chunks is not None and len(docset.table_chunks) > 0:
                                print(f"✅ table_chunks: 有 {len(docset.table_chunks)} 个元素")
                                for i, chunk in enumerate(docset.table_chunks[:3]):  # 只显示前3个
                                    print(f"  Table {i+1}: {str(chunk)[:50]}...")
                            else:
                                print("❌ table_chunks: 为空或None")
                            
                            # 检查 text_chunks
                            if docset.text_chunks is not None and len(docset.text_chunks) > 0:
                                print(f"✅ text_chunks: 有 {len(docset.text_chunks)} 个元素")
                                for i, chunk in enumerate(docset.text_chunks[:3]):  # 只显示前3个
                                    print(f"  Text {i+1}: {str(chunk)[:50]}...")
                            else:
                                print("❌ text_chunks: 为空或None")
                            
                            print("=" * 50)
        
        '''self.pdf_parser_helper.docs = self.docs
        self.pdf_parser_helper.remain_docparser()
        self.docs = self.pdf_parser_helper.docs'''

        #self.serialize_docs()
                   
    def serialize_docs(self):
        """
        Serialize the extracted documents into JSON files.
        """
        output_dir = self.json_path
        for doc in self.docs:
            with open(self.arxiv_pool, "a", encoding="utf-8") as f:
                f.write(doc.doc_id+'\n')
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                doc_dict = doc.model_dump()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)