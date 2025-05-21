from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from uuid import uuid4
from pathlib import Path
#new
from datetime import datetime, timezone, date, timedelta
import arxiv
import os
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time


'''
TODO 
1.change the init_docset and store the metadata in the docset dirctly
2.store the raw html in a folder and delete the url_folder.
3.change test function
4.add pdf_extractor if html dont work
'''

class ArxivHTMLExtractor():
    """
    A class used to extract information from daily arXiv HTMLs and serialize it into JSON files.
    """
    def __init__(self, html_text_folder, pdf_folder_path, arxiv_pool, image_folder_path,json_path):
        self.date = datetime.now(timezone.utc).date()
        self.docs = []
        self.html_text_folder = html_text_folder
        self.pdf_folder_path = pdf_folder_path
        self.arxiv_pool = arxiv_pool
        self.image_folder_path = image_folder_path
        self.json_path = json_path

###################################################   Search papers and Update Docset’s metadata part    ############################################################
    
    def init_docset(self):
        client = arxiv.Client()
        one_day = timedelta(days=1)
        yesterday = self.date - one_day

        exact_time = "0600"
        today_str = self.date.strftime("%Y%m%d") + exact_time
        yesterday_str = yesterday.strftime("%Y%m%d") + exact_time
        print(today_str)
        #query = "cat:cs.* AND submittedDate:[" + yesterday_str + " TO " + today_str + "]"
        query = "cat:cs.* AND submittedDate:[202505190600 TO 202505200600]"
        print(today_str,yesterday_str)

        search = arxiv.Search(
            query=query,
            max_results=5,  # You can set max papers you want here
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        #print(f"grabbing arXiv papers in cs.* submitted from {yesterday} to {self.date}......")
        print(f"grabbing arXiv papers in cs.* submitted from 202505190600 to 202505200600......")

        for result in client.results(search):
            time.sleep(3)
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
                pdf_path=self.download_arxiv_pdf(arxiv_id, self.pdf_folder_path),
                #Set htmlpath to None first and update it later
                HTML_path=None 
            )

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

            with open(self.arxiv_pool, "a", encoding="utf-8") as f:
                f.write(arxiv_id+'\n')

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

#########################################################   Update Docset’s chunk part   ####################################################################


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
        #for fig in soup.find_all('figure'):
        for fig in soup.find_all(lambda tag: tag.name == 'figure' and 'ltx_table' not in tag.get('class', [])):
            img = fig.find('img')
            caption = fig.find('figcaption')
            fig_id = fig.get('id', '')

            if img and caption:
                tag = caption.find('span', class_='ltx_tag_figure')
                if tag:
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
                    img_url = urljoin(f"https://arxiv.org/html/{arxivid}/", img_src) 
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
    
                    
#########################################################   Total update dataset function   ####################################################################

    def extract_all_htmls(self, img_path = "", pdf_path = "") -> DocSet:

        self.init_docset()

        for filename in os.listdir(self.html_text_folder):
            if filename.endswith(".html"):
                file_path = os.path.join(self.html_text_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    soup = BeautifulSoup(html_content, "html.parser")

                    for docset in self.docs:
                        if docset.doc_id == filename[:-5] and docset.HTML_path is not None:
                            figurechunks = self.extract_figures_to_folder(soup,self.image_folder_path,docset.doc_id)
                            table_chunks = self.extract_tables(soup,docset.doc_id)
                            docset.text_chunks = self.extract_text(soup)
                            docset.figure_chunks = figurechunks
                            docset.table_chunks = table_chunks

        self.serialize_docs()


    def extract_remain_papers_using_pdf(self, img_path = "", pdf_path = "") -> DocSet:
        pass
                        
    def serialize_docs(self):
        """
        Serialize the extracted documents into JSON files.
        """
        output_dir = self.json_path
        for doc in self.docs:
            output_path = Path(output_dir) / f"{doc.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                doc_dict = doc.model_dump()
                json_str = json.dumps(doc_dict, indent=4)
                f.write(json_str)

if __name__ == '__main__':
    html_text_folder = "/data3/peirongcan/paperIgnite/AIgnite/test/htmls"
    pdf_folder = "/data3/peirongcan/paperIgnite/AIgnite/test/pdfs"
    arxiv_pool = "/data3/peirongcan/paperIgnite/AIgnite/test/html_url_storage/html_urls.txt"
    image_folder_path = "/data3/peirongcan/paperIgnite/AIgnite/test/imgs"
    json_path = "/data3/peirongcan/paperIgnite/AIgnite/test/jsons"
    extractor = ArxivHTMLExtractor(html_text_folder,pdf_folder,arxiv_pool,image_folder_path,json_path)
    extractor.extract_all_htmls()