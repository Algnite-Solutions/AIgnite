from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk
from uuid import uuid4
from pathlib import Path
#new
import arxiv
import os
import requests
from urllib.parse import urljoin

class ArxivHTMLExtractor():
    #TODO: @rongcan, finish this extracter and serialize into a json
    def __init__(self):
        self.docs = []

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
        
    def get_published_date(self, arxiv_id: str) -> str | None:
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
        
    def extract_figures(self, soup):
        figures = []
        for fig in soup.find_all('figure'):
            img = fig.find('img')
            caption = fig.find('figcaption')
            fig_id = fig.get('id', '')

            if img and caption:
                tag = caption.find('span', class_='ltx_tag_figure')
                figure_name = tag.text.strip().rstrip(':').strip()
                figure_name = figure_name.replace(' ', '')  # 把所有空格去掉

                img_src = img['src']
                img_url = urljoin("https://ar5iv.labs.arxiv.org/", img_src)  # 获取完整的图片 URL
                img_data = requests.get(img_url).content
                alt = img.get('alt', '')
                caption_text = caption.get_text(strip=True)
                figures.append({
                    'name':figure_name,
                    'id': fig_id,
                    'img_src': img_url,
                    'alt': alt,
                    'caption': caption_text,
                    'img_data': img_data #图片本身数据
                })
        return figures

    def extract_figures_to_folder(self, soup, img_path):
        figures = []
        for fig in soup.find_all('figure'):
            img = fig.find('img')
            caption = fig.find('figcaption')
            fig_id = fig.get('id', '')

            if img and caption:
                tag = caption.find('span', class_='ltx_tag_figure')
                figure_name = tag.text.strip().rstrip(':').strip()
                figure_name = figure_name.replace(' ', '')  #Remove all the Spaces

                img_src = img['src']
                img_url = urljoin("https://ar5iv.labs.arxiv.org/", img_src)  #Get the complete image URL
                alt = img.get('alt', '')
                caption_text = caption.get_text(strip=True)
                img_data = requests.get(img_url).content
                img_filename = os.path.join(img_path, f'{figure_name}.png')  # The file name of the stored picture

                #Make sure the image storage directory exists
                os.makedirs(os.path.dirname(img_filename), exist_ok=True)

                #Save the picture to the local machine
                with open(img_filename, 'wb') as f:
                    f.write(img_data)
                
                figures.append({
                    'name':figure_name,
                    'id': fig_id,
                    'img_src': img_url,
                    'alt': alt,
                    'caption': caption_text,
                    'local_img_path': img_filename  # The local path for saving the picture
                })

        return figures

    def extract_docset(self, html: str) -> DocSet:
        soup = BeautifulSoup(html, "html.parser")

        title = self.extract_title(soup)
        #print(title)
        abstract = self.extract_abstract(soup)
        #print(abstract)
        arxiv_id = self.extract_arxiv_id(soup)
        #print(arxiv_id)
        authors = self.extract_authors(soup)
        #print(authors)
        published_date = self.get_published_date(arxiv_id)
        #print(published_date)
        categories = self.get_categories(arxiv_id)
        #print(categories)
        #imgs = self.extract_figures(soup)
        #print(imgs)

        text_chunks = [TextChunk(id=str(uuid4()), type="text", text=abstract)]
        

        self.docs.append(DocSet(
            doc_id = arxiv_id,
            title = title,
            authors = authors,
            categories = categories,
            published_date = published_date,
            abstract = abstract,
            chunks = text_chunks,
        ))
    
    def serialize_docs(self, output_dir: str):
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.json(indent=4))

if __name__ == "__main__":
    extractor = ArxivHTMLExtractor()
    html = extractor.load_html("/app/test/html_doc/test.txt")
    extractor.extract_docset(html)
    