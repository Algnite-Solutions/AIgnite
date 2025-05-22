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
import google.generativeai as genai
import re
from urllib.parse import urlparse
from typing import List, Tuple
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from volcengine.visual.VisualService import VisualService
import json

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
        self.pdf_parser_helper = ArxivPDFExtractor(self.docs, pdf_folder_path, image_folder_path)

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
        query = "cat:cs.* AND submittedDate:[202504190600 TO 202504200600]"
        print(today_str,yesterday_str)

        search = arxiv.Search(
            query=query,
            max_results=1,  # You can set max papers you want here
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        print(f"grabbing arXiv papers in cs.* submitted from {yesterday} to {self.date}......")
        #print(f"grabbing arXiv papers in cs.* submitted from 202504190600 to 202504200600......")

        for result in client.results(search):
            time.sleep(15)
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
                pdf_path=download_arxiv_pdf(arxiv_id, self.pdf_folder_path),
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

        self.serialize_docs_init()


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
                            docset.figure_chunks = figurechunks
                            docset.table_chunks = table_chunks
                            docset.text_chunks = self.extract_text(soup)
        
        self.pdf_parser_helper.docs = self.docs
        self.pdf_parser_helper.remain_docparser()
        self.docs = self.pdf_parser_helper.docs

        self.serialize_docs()

    def extract_remain_papers_using_pdf(self, img_path = "", pdf_path = "") -> DocSet:
        pass
                        
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

    def serialize_docs_init(self):
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


############################################################### Some Tools ####################################################################

'''def download_arxiv_pdf(arxiv_id: str, save_path):
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
'''

def download_arxiv_pdf(arxiv_id: str, save_path):
    """
    Download the PDF file on arXiv to the specified path with retry logic.
    Parameter: arxiv_id (str): The arXiv ID of the paper, such as '2106.14834' 
    save_path (str): Local file path to save the PDF folder
    """
    max_retries = 5
    retry_count = 0
    url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

    while retry_count < max_retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 检查HTTP响应状态码
            save_path = os.path.join(save_path, f'{arxiv_id}.pdf')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"PDF successfully saved to: {save_path}")
            return save_path

        except requests.HTTPError as e:
            error_code = response.status_code
            print(f"Download failed (HTTP {error_code}), retrying...")
        except Exception as e:
            print(f"Download failed: {str(e)}, retrying...")

        retry_count += 1
        if retry_count < max_retries:
            print(f"Retrying in 15 seconds (Attempt {retry_count}/{max_retries})")
            time.sleep(15)  # 等待15秒后重试

    print(f"Failed to download after {max_retries} attempts.")
    return None

def get_img_from_url(arxivid,img_src):
    time.sleep(1)
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIgniteBot/1.0; +https://github.com/Algnite-Solutions/AIgnite)"
    }

    img_url = urljoin(f"https://arxiv.org/html/{str(arxivid)}/", img_src)
    #print(f"[INFO] Fetching image from: {img_url}")

    try:
        response = session.get(img_url, headers=headers, timeout=10)
        response.raise_for_status()
        img_data = response.content
        return img_data
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch image {img_url}: {e}")
        return None

def get_pdf_md(path,store_path,name):
    visual_service = VisualService()
    # call below method if you dont set ak and sk in $HOME/.volc/config
    visual_service.set_ak('') attention!
    visual_service.set_sk('')

    params = dict()

    form = {
        "image_base64":  base64.b64encode(open(str(path),'rb').read()).decode(),   # 文件binary 图片/PDF 
        "image_url": "",                  # url
        "version": "v3",                  # 版本
        "page_start": 0,                  # 起始页数
        "page_num": 16,                   # 解析页数
        "table_mode": "html",             # 表格解析模式
        "filter_header": "true"           # 过滤页眉页脚水印
    }

    # 请求
    resp = visual_service.ocr_pdf(form)

    if resp["data"]:
        markdown = resp["data"]["markdown"] # markdown 字符串
        #json_data = resp["data"]["detail"] # json格式详细信息

        # 确保目录存在
        os.makedirs(store_path, exist_ok=True)
        # 完整文件路径
        file_path = os.path.join(store_path, f"{name}.md")

        # 写入文件
        with open(file_path, "w") as f:
            f.writelines(markdown)

        #json_data = json.loads(json_data)

    else:
        print("request error")

    return file_path

def get_Gemini_response(api_key,file_path,prompt):

    os.environ['http proxy']="http://127.0.0.1:7890"
    os.environ['https proxy']="http://127.0.0.1:7890"

    genai.configure(api_key=api_key)

    #uploaded_file = genai.upload_file(path="/data3/peirongcan/paperIgnite/AIgnite/src/AIgnite/data/resp.md", display_name="Sample PDF")
    uploaded_file = genai.upload_file(path = file_path, display_name="Sample")
    print("Uploaded file name:", uploaded_file.name)

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content([
        uploaded_file,
        prompt
    ])

    return(response.text)

def _parse_image_urls(content: str,arxiv_id) -> List[Tuple[str, str, str]]:
    """内部函数：解析图片URL并提取符合规则的名称"""
    # 正则规则说明：
    # 1. 匹配![任意内容](URL) 格式
    # 2. 提取名称：在`)`之后，寻找以F/f开头、第一个数字结尾的字符串
    pattern = r'!\[.*?\]\((https?://[^\)]+)\)'  # 先提取所有图片链接
    all_matches = re.findall(pattern, content)
    
    image_info = []
    for url in all_matches:
        # 从URL前后的上下文中提取名称（假设名称在`)`之后，格式为F/f开头+数字结尾）
        # 示例：`) Fig_1"` 或 `) fig3. `
        name = _extract_name_from_context(content, url)
        name = arxiv_id+'_'+name
        caption = _extract_caption_from_context(content, url)
        if name:
            image_info.append((name, url, caption))
    return image_info

def _extract_name_from_context(content: str, url: str) -> str:
    """从URL后的文本中提取以F开头、数字结尾的名称，去除点和空格"""
    # 定位URL在文本中的位置（查找URL后的内容）
    url_end = content.find(url) + len(url)
    post_url_content = content[url_end:].strip()  # 获取URL之后的文本
    
    # 正则匹配：以F开头，任意字符（排除点和空格），以数字结尾
    pattern = r'(fig(?:ure)?\.?\s*\d+)'
    match = re.search(pattern, post_url_content, re.IGNORECASE)  # 不区分大小写
    
    if match:
        # 提取匹配内容并去除点和空格
        raw_name = match.group(0)
        cleaned_name = raw_name.replace('.', '').replace(' ', '')  # 去除点和空格
        return cleaned_name.lower() if cleaned_name.startswith('f') else cleaned_name  # 统一小写（可选）
    return ""

def _extract_caption_from_context(content: str, url: str) -> str:
    """从 URL 后提取图注文本，直到下一个换行符"""
    url_end = content.find(url) + len(url)
    post_url_content = content[url_end:].lstrip()

    # 提取从 'Fig' 开头到换行符结束的一整行作为 caption
    match = re.search(r'(fig(?:ure)?\.?\s*\d+[a-zA-Z]?\.*.*?)\n', post_url_content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def _download_single_image(name: str, url: str, save_dir: str) -> bool:
    """内部函数：下载单张图片"""
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        # 处理文件扩展名（支持URL中带查询参数的情况）
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            ext = 'jpg'  # 默认扩展名
        
        file_name = f"{name}.{ext}"
        save_path = os.path.join(save_dir, file_name)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 成功下载：{name} -> {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败 {url}：{str(e)}")
        return False


############################################### PDF #########################################################

class ArxivPDFExtractor():
    def __init__(self, docs, pdf_folder_path, image_folder_path, arxiv_pool, json_path):
        if docs is None:
            self.docs = []
        else:
            self.docs = docs
        self.pdf_folder_path = pdf_folder_path
        self.image_folder_path = image_folder_path
        self.pdf_paths = []
        self.arxiv_pool = arxiv_pool
        self.json_path = json_path
        self.date = datetime.now(timezone.utc).date()

    def init_docset(self):
        client = arxiv.Client()
        one_day = timedelta(days=1)
        yesterday = self.date - one_day

        exact_time = "0600"
        today_str = self.date.strftime("%Y%m%d") + exact_time
        yesterday_str = yesterday.strftime("%Y%m%d") + exact_time
        print(today_str)
        #query = "cat:cs.* AND submittedDate:[" + yesterday_str + " TO " + today_str + "]"
        query = "cat:cs.* AND submittedDate:[202504190600 TO 202504200600]"
        print(today_str,yesterday_str)

        search = arxiv.Search(
            query=query,
            max_results=5,  # You can set max papers you want here
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        print(f"grabbing arXiv papers in cs.* submitted from {yesterday} to {self.date}......")
        #print(f"grabbing arXiv papers in cs.* submitted from 202504190600 to 202504200600......")

        for result in client.results(search):
            time.sleep(5)
            arxiv_id = result.pdf_url.split('/')[-1]
            with open(self.arxiv_pool, "r", encoding="utf-8") as f:
                if arxiv_id in f.read():
                    print(f"{arxiv_id} is already extracted before!")
                    continue
            #add basic info
            add_doc = DocSet(
            doc_id=arxiv_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            categories=result.categories,
            published_date=str(result.published),
            abstract=result.summary,
            pdf_path=download_arxiv_pdf(arxiv_id, self.pdf_folder_path),
            #Set htmlpath to None first and update it later
            HTML_path=None )

            self.docs.append(add_doc)

        #self.serialize_docs_init()

    def extract_all(self):
        self.init_docset()
        for doc in self.docs:
            path = doc.pdf_path
            print("getting markdown...")
            markdown_path = get_pdf_md(path,self.pdf_folder_path,doc.doc_id)
            print("done")
            doc.figure_chunks = self.pdf_images_chunk(markdown_path,self.image_folder_path,doc.doc_id)
            doc.table_chunks = self.pdf_tables_chunk(markdown_path)
            doc.text_chunks = self.pdf_text_chunk(markdown_path)#一定在最后
        self.serialize_docs()

    def remain_docparser(self):
        for doc in self.docs:
            if doc.HTML_path == None:
                self.pdf_paths.append(str(Path(self.pdf_folder_path) / f"{doc.doc_id}.pdf"))
                print(self.pdf_paths)

                for path in self.pdf_paths:
                    print("getting markdown...")
                    markdown_path = get_pdf_md(path,self.pdf_folder_path,doc.doc_id)
                    print("done")
                    doc.figure_chunks = self.pdf_images_chunk(markdown_path,self.image_folder_path,doc.doc_id)
                    doc.table_chunks = self.pdf_tables_chunk(markdown_path)
                    doc.text_chunks = self.pdf_text_chunk(markdown_path)#一定在最后
                   
    def pdf_images_chunk(self, markdown_path, image_folder_path, doc_id):
        figures = []

        try:
            # 读取Markdown内容
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                
            # 解析图片信息
            image_list = _parse_image_urls(md_content, doc_id)
            if not image_list:
                print("警告：未在Markdown中发现图片链接")
                return
                
            # 创建保存目录
            os.makedirs(image_folder_path, exist_ok=True)
            
            # 执行下载
            success_count = 0
            for name, url, caption in image_list:
                if _download_single_image(name, url, image_folder_path):
                    success_count += 1
                    figures.append(FigureChunk(
                        id = None,
                        title = name,
                        type = ChunkType.FIGURE,
                        image_path = str(os.path.join(image_folder_path, name)),
                        alt_text = "Refer to caption",
                        caption = caption
                    ))
            
            print(f"\n📌 下载完成：共处理{len(image_list)}张图片，成功保存{success_count}张")
            
        except FileNotFoundError:
            print(f"错误：Markdown文件未找到 - {markdown_path}")
            
        except Exception as e:
            print(f"程序异常：{str(e)}")
        
        #print(figures)
        return figures
        

    def pdf_tables_chunk(self, markdown_path):
        tables = []
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            soup = BeautifulSoup(md_content, 'html.parser')
            all_tables = soup.find_all('table')

            for idx, table in enumerate(all_tables):
                table_html = str(table)

                # 在 Markdown 内容中查找该表格的文本位置
                table_pos = md_content.find(table_html)
                context_before = md_content[max(0, table_pos - 500):table_pos]

                # 从前文中找 Table 标题
                caption_match = re.search(r'(Table\s*\d+[.:]?\s*)([^\n<]+)', context_before, re.IGNORECASE)
                if caption_match:
                    table_name = caption_match.group(1).strip().replace(':', '').replace('.', '')
                    caption_text = caption_match.group(2).strip()
                else:
                    table_name = f'table_{idx+1}'
                    caption_text = ''

                tables.append(TableChunk(
                    id=None,
                    title=table_name,
                    type=ChunkType.TABLE,
                    table_html=table_html,
                    caption=caption_text
                ))
                

        except FileNotFoundError:
            print(f"错误：Markdown文件未找到 - {markdown_path}")
        except Exception as e:
            print(f"程序异常：{str(e)}")
        #print(tables)
        return tables
    
    def pdf_text_chunk(self, markdown_path) -> List[TextChunk]:
        all_text = []

        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            md_content = re.sub(r'^!\[fig_[^\n]*\n?', '', md_content, flags=re.MULTILINE)

            # 查找所有一级标题（## 开头，排除如 2.1 开头的子标题）
            # pattern = r'(?:^|\n)(##\s+(?!\d+\.)[^\n]+)'
            pattern = r'(?:^|\n)(##\s+(?![A-Za-z0-9]+\.)[^\n]+)'
            matches = list(re.finditer(pattern, md_content))

            # 为方便处理，记录所有段落起始位置
            section_boundaries = [m.start() for m in matches]
            section_boundaries.append(len(md_content))  # 加入最后的终点

            for i in range(len(matches)):
                start = section_boundaries[i]
                end = section_boundaries[i + 1]
                section_text = md_content[start:end].strip()

                header_line = matches[i].group(1).strip()
                title = header_line.lstrip('#').strip()

                section_id = f"text_{i+1}"

                all_text.append(TextChunk(
                    id=section_id,
                    type=ChunkType.TEXT,
                    title=title,
                    caption=title,
                    text=section_text
                ))

        except FileNotFoundError:
            print(f"错误：Markdown文件未找到 - {markdown_path}")
        except Exception as e:
            print(f"程序异常：{str(e)}")
        return all_text
    
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


if __name__ == '__main__':
    os.environ['http proxy']="http://127.0.0.1:7890"
    os.environ['https proxy']="http://127.0.0.1:7890"
    html_text_folder = "/data3/peirongcan/paperIgnite/AIgnite/test/htmls"
    pdf_folder = "/data3/peirongcan/paperIgnite/AIgnite/test/pdfs"
    arxiv_pool = "/data3/peirongcan/paperIgnite/AIgnite/test/html_url_storage/html_urls.txt"
    image_folder_path = "/data3/peirongcan/paperIgnite/AIgnite/test/imgs"
    json_path = "/data3/peirongcan/paperIgnite/AIgnite/test/jsons"
    #extractor = ArxivHTMLExtractor(html_text_folder,pdf_folder,arxiv_pool,image_folder_path,json_path)
    #extractor.extract_all_htmls()
    extractor2 = ArxivPDFExtractor(None, pdf_folder, image_folder_path, arxiv_pool, json_path)
    extractor2.extract_all()
    #extractor2.pdf_text_chunk('/data3/peirongcan/paperIgnite/AIgnite/test/pdfs/2505.15817v1.md') 
    #extractor.extract_all_htmls()
    #download_images_from_markdown("/data3/peirongcan/paperIgnite/AIgnite/test/pdfs/2505.13959v1.md", "/data3/peirongcan/paperIgnite/AIgnite/test/imgs")

