from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from uuid import uuid4
from pathlib import Path
#new
from google import genai
from google.genai import types
import arxiv
import os
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

class ArxivHTMLExtractor():
    """
    A class used to extract information from arXiv HTML pages and serialize it into JSON files.
    """
    def __init__(self):
        self.client = arxiv.Client()
        self.docs = []

    def download_html(self, url: str, source: str) -> str:
        """
        Args:
            url (str): The URL of the HTML page to download, must start with "https://ar5iv.labs.arxiv.org/html/".
            source (str): The directory path to save the downloaded file.
        Returns:
            str: The path of the saved HTML file, or None if the download fails.
        """
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
        """
        Args:
            source (str): The path to the file containing the HTML content.
        Returns:
            str: The HTML content from the file.
        """
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
    
    def get_published_date(self, arxiv_id: str):
        "Import Arxiv to search online for the published date of a paper."
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = self.client.results(search)
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
        "Import Arxiv to search online for the categories of a paper."
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = self.client.results(search)
            paper = next(results, None)
            if paper is None:
                print(f"Warning：paper with arXiv ID '{arxiv_id}' NOT FOUND")
                return []
            return paper.categories
        except Exception as e:
            print(f"Error when extracting categories: WRONG ID or other errors - {str(e)}")
            return []  

    def extract_figures_to_folder(self, soup, img_path):
        """
        Extract figures from a BeautifulSoup object and save them to a folder.
        Args:
            soup (BeautifulSoup): A BeautifulSoup object containing the HTML content.
            img_path (str): The path to the folder where the figures will be saved.
        Returns:
            list: A list containing FigureChunk objects, each representing a figure, or an empty list if no figures are found.
        """
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
        """
        Extract tables from a BeautifulSoup object.
        Args:
            soup (BeautifulSoup): A BeautifulSoup object containing the HTML content.
        Returns:
            list: A list containing TableChunk objects, each representing a table, or an empty list if no tables are found.
        """
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
    
    def download_arxiv_pdf(self, arxiv_id: str, save_path):
        """
        Download the PDF file on arXiv to the specified path. 
        Parameter: arxiv_id (str): The arXiv ID of the paper, such as '2106.14834' 
        save_path (str): to save the local file path of the PDF folder
        """
        url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

        try:
            response = requests.get(url, stream=True)
            #Check whether the download was successful
            response.raise_for_status()  
            save_path = os.path.join(save_path, f'{arxiv_id}.pdf')

            # Make sure the directory exists.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"The PDF has been successfully saved to：{save_path}")
        except requests.HTTPError as e:
            print(f"Download failed, HTTP error{e}")
        except Exception as e:
            print(f"Download failed. Error message:{e}")

        return save_path

    def extract_docset(self, html: str, img_path, pdf_path) -> DocSet:
        """
        Extract a document set from HTML content.
        Args:
            html (str): The HTML content to extract information from.
            img_path (str): The path to the folder where the figures will be saved.
        Returns:
            DocSet: A DocSet object containing the extracted information.
        """
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

        print("downloading pdf...")
        pdf_path = self.download_arxiv_pdf(arxiv_id, pdf_path)
        #print(pdf_path)

        ################################chunks######################################
        print("tables extracting...")
        tables = self.extract_tables(soup)
        print("figures extracting...")
        figures = self.extract_figures_to_folder(soup,img_path = img_path)
        print("text extracting...")
        text = self.extract_text(soup)# This must be placed last
        ############################################################################
        
        add_doc = DocSet(
            doc_id = arxiv_id,
            title = title,
            authors = authors,
            categories = categories,
            published_date = published_date,
            abstract = abstract,
            text_chunks = text,
            figure_chunks = figures,
            table_chunks = tables,
            pdf_path = pdf_path 
        )
        self.docs.append(add_doc)
        return add_doc
    
    def serialize_docs(self, output_dir: str):
        """
        Serialize the extracted documents into JSON files.
        """
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                doc_dict = doc.model_dump()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)

    def __del__(self):
        del self.client

class ArxivLaTeXExtractor():
    def __init__(self, latex_floder_path):
        self.latex_path = latex_floder_path
        self.docs = []

    def get_imgs(self,output_img_path):
        pass

    def get_tables(self):
        pass

    def serialize_docs(self, output_dir: str):
        """
        Serialize the extracted documents into JSON files.
        """
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                doc_dict = doc.model_dump()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)

class ArxivPDFExtractor():
    def __init__(self):
        self.docs = []

    def get_imgs(self,output_img_path):
        # Create a client
        os.environ['http_proxy'] = "http://127.0.0.1:7890"
        os.environ['https_proxy'] = "http://127.0.0.1:7890"
        api_key = "AIzaSyDQS4jFfedzDourgwQxiP4hhOR0lK67l44"
        client = genai.Client(api_key=api_key)
        
        model_id =  "gemini-2.0-flash-preview-image-generation" # or "gemini-2.0-flash-lite-preview-02-05"  , "gemini-2.0-pro-exp-02-05"

        my_test_pdf = client.files.upload(file="/data3/peirongcan/paperIgnite/AIgnite/test/pdf_imgs/a0ff53f39178ce535bb430ee1f5d0bd.png", config={'display_name': 'test'})
        file_size = client.models.count_tokens(model=model_id,contents=my_test_pdf)
        print(f'File: {my_test_pdf.display_name} equals to {file_size.total_tokens} tokens')

        prompt = f"I gave you a image named my_test_pdf. This is one page of a paper, Please return the figure in the paper. Please use the cropping function to cut out part of the picture instead of generating a new one"
        response = client.models.generate_content(model=model_id, contents=[prompt, my_test_pdf], config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE']))
        # Convert the response to the pydantic model and return it
        print(type(response))
        #print(response)
        # 假设 response 是你拿到的响应对象

        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type == "image/png":
                image_data = part.inline_data.data
                break

        if image_data:
            with open("/data3/peirongcan/paperIgnite/AIgnite/test/pdf_imgs/image.jpg", "wb") as f:
                f.write(image_data)
            print("图像已保存为image.jpg")
        else:
            print("没有找到图像数据")

    def get_tables(self):
        pass

    def serialize_docs(self, output_dir: str):
        """
        Serialize the extracted documents into JSON files.
        """
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                doc_dict = doc.model_dump()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)

if __name__ == '__main__':
    pdf_extractor = ArxivPDFExtractor()
    pdf_extractor.get_imgs("/data3/peirongcan/paperIgnite/AIgnite/test/pdf_imgs")