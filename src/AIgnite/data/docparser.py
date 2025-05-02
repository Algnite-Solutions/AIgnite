from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from uuid import uuid4
from pathlib import Path
#new
import arxiv
import os
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

class ArxivHTMLExtractor():
    #TODO: @rongcan, finish this extracter and serialize into a json
    def __init__(self):
        self.docs = []


    def download_html(self, url: str, source: str) -> str:
        #设置三次最大重试次数
        max_retries = 3
        assert url.startswith("https://ar5iv.labs.arxiv.org/html/"), f"URL {url} must begin with https://ar5iv.labs.arxiv.org/html/"
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                html_content = response.text

                if not os.path.exists(source):
                    os.makedirs(source)

                # Generate the file name. Here simply use the last part of the URL as the file name
                file_name = os.path.join(source, url.split("/")[-1] + ".html")

                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(html_content)
                print(f"The web page content has been successfully saved to {file_name}")
                return file_name
            except requests.RequestException as e:
                print(f"An error occurred in the request: {e}. Retrying...")
                retries += 1
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying...")
                retries += 1

        print("Max retries reached. Download failed.")
        return None


    def load_html(self, source: str) -> str:
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
    
    def extract_title(self, soup: BeautifulSoup):
        try:
            title_tag = soup.find("h1")
            if title_tag:
                return title_tag.text.strip()
            return None
        except Exception as e:
            print(f"Error when extracting title: {e}")
            return None

    def extract_abstract(self, soup: BeautifulSoup):
        try:
            abstract_div = soup.find("div", class_="ltx_abstract")
            if abstract_div:
                abstract_p = abstract_div.find("p")
                if abstract_p:
                    return abstract_p.get_text(strip=False)
            return None
        except Exception as e:
            print(f"Error when extracting abstract: {e}")
            return None
        
    def extract_arxiv_id(self, soup: BeautifulSoup):
        try:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.text
                start_index = title_text.find('[')
                end_index = title_text.find(']')
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    arxiv_id = title_text[start_index + 1: end_index]
                    return arxiv_id
            return None
        except Exception as e:
            print(f"Error when extracting arxiv id: {e}")
            return None
        
    def extract_authors(self, soup: BeautifulSoup):
        try:
            author_spans = soup.find_all("span", class_="ltx_personname")
            authors = []
            for span in author_spans:
                text = span.get_text(strip=True)
                if text:
                    authors.extend(text.split(","))
            return [author.strip() for author in authors]
        except Exception as e:
            print(f"Error when extracting authors: {e}")
            return []
    
    def extract_text(self, soup: BeautifulSoup):
        try:
            article = soup.find('article')
            all_text = []
            sections = article.find_all('section', class_ = ['ltx_section', 'ltx_appendix'])
            if sections:
                for section in sections:
                    # 移除figure标签及其内容
                    for figure in section.find_all('figure'):
                        figure.extract()
                    section_text = section.get_text()
                    section_text = section_text.replace('\n\n', '\n')
                    # 获取section的id
                    section_id = section.get('id', '')  
                    title_elem = section.find('h2', class_='ltx_title ltx_title_section')
                    # 获取h2section标题
                    title = title_elem.get_text(strip=True) if title_elem else ''  

                    # 这里根据ar5iv的html结构，没有明显可作为caption的内容，先设为title
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
    
    def get_published_date(self, arxiv_id: str):
        "Import Arxiv to search online."
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = client.results(search)
            paper = next(results, None)
            if paper is None:
                print(f"Warning：paper with arXiv ID '{arxiv_id}' NOT FOUND")
                return None
            # Extract the release date and format it as a string
            published_date = paper.published.strftime("%Y-%m-%d")
            return published_date
        except Exception as e:
            print(f"Error when extracting authors:WRONG ID or other errors - {str(e)}")
            return None
        
    def get_categories(self, arxiv_id: str):
        "Import Arxiv to search online."
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = client.results(search)
            paper = next(results, None)
            if paper is None:
                print(f"Warning：paper with arXiv ID '{arxiv_id}' NOT FOUND")
                return []
            return paper.categories
        except Exception as e:
            print(f"Error when extracting categories: WRONG ID or other errors - {str(e)}")
            return []  

    def extract_figures_to_folder(self, soup, img_path):
        figures = []
        arxivid = self.extract_arxiv_id(soup)
        for fig in soup.find_all('figure'):
            img = fig.find('img')
            caption = fig.find('figcaption')
            fig_id = fig.get('id', '')

            if img and caption:
                tag = caption.find('span', class_='ltx_tag_figure')
                figure_name = tag.text.strip().rstrip(':').strip()
                 #Remove all the Spaces
                figure_name = figure_name.replace(' ', '') 
                if figure_name.endswith('.'):
                    figure_name = figure_name[:-1]
                if not figure_name.startswith("Figure"):
                    figure_name = "Figure" + fig_id[4] + figure_name
                figure_name = str(arxivid)+'_'+figure_name


                img_src = img['src']
                #Get the complete image URL
                img_url = urljoin("https://ar5iv.labs.arxiv.org/", img_src) 
                alt = img.get('alt', '')
                caption_text = caption.get_text(strip=True)
                img_data = requests.get(img_url).content
                #The file name of the stored picture
                img_filename = os.path.join(img_path, f'{figure_name}.png')

                #Make sure the image storage directory exists
                os.makedirs(os.path.dirname(img_filename), exist_ok=True)

                #Save the picture to the local machine
                with open(img_filename, 'wb') as f:
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
    
    def extract_tables(self, soup):
        tables = []
        arxivid = self.extract_arxiv_id(soup)
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

    def extract_docset(self, html: str, img_path) -> DocSet:
        soup = BeautifulSoup(html, "html.parser")

        print("title extracting...")
        title = self.extract_title(soup)
        #print(title)

        print("abstract extracting...")
        abstract = self.extract_abstract(soup)
        #print(abstract)

        print("id extracting...")
        arxiv_id = self.extract_arxiv_id(soup)
        #print(arxiv_id)

        print("authors extracting...")
        authors = self.extract_authors(soup)
        #print(authors)

        print("date extracting...")
        published_date = self.get_published_date(arxiv_id)
        #print(published_date)

        print("categories extracting...")
        categories = self.get_categories(arxiv_id)
        #print(categories)

        ################################chunks######################################
        print("tables extracting...")
        tables = self.extract_tables(soup)
        print("figures extracting...")
        figures = self.extract_figures_to_folder(soup,img_path = img_path)
        print("text extracting...")
        text = self.extract_text(soup)#这个一定要放最后
        ############################################################################
        
        self.docs.append(DocSet(
            doc_id = arxiv_id,
            title = title,
            authors = authors,
            categories = categories,
            published_date = published_date,
            abstract = abstract,
            text_chunks = text,
            figure_chunks = figures,
            table_chunks = tables
        ))

        return self.docs
    
    '''def serialize_docs(self, output_dir: str):
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.json(indent=4))'''
    def serialize_docs(self, output_dir: str):
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                # 将 pydantic 模型转换为字典，再用 json.dumps 转换为 JSON 字符串
                doc_dict = doc.dict()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)