from bs4 import BeautifulSoup
from .docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from pathlib import Path
#new
import fitz  # PyMuPDF
import os
import io
from datetime import datetime, timezone
import arxiv
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
from abc import ABC, abstractmethod

class BasePDFExtractor(ABC):
    """Abstract base classes of the PDF extractor"""

    def __init__(self, docs, pdf_folder_path, image_folder_path, arxiv_pool, json_path, volcengine_ak, volcengine_sk, start_time, end_time):
        """
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
        """
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
        self.ak = volcengine_ak
        self.sk = volcengine_sk
        self.start_time = start_time
        self.end_time = end_time

    @abstractmethod
    def extract_all(self):
        """Carry out the complete extraction process"""
        pass

    @abstractmethod
    def remain_docparser(self):
        """provide help for HTMLparser"""
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
    def pdf_text_chunk(self, markdown_path):
        """Extract the text chunk from Markdown produced by PDF"""
        pass

    @abstractmethod
    def pdf_images_chunk(self, markdown_path, image_folder_path, doc_id):
        """Extract the img chunk from Markdown produced by PDF"""
        pass

    @abstractmethod
    def pdf_tables_chunk(self, markdown_path):
        """Extract the table chunk from Markdown produced by PDF"""
        pass

class ArxivPDFExtractor(BasePDFExtractor):
    """
    A class used to extract information from daily arXiv PDFs and serialize it into JSON files.
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
            max_results=None,  # You can set max papers you want here
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        print(f"grabbing arXiv papers in cs.* submitted from {self.start_time} to {self.end_time}......")

        tem = client.results(search)
        tem = list(tem)
        print("successful search!")
        for result in tem:
            print(1)
            arxiv_id = result.pdf_url.split('/')[-1]
            print(2)
            with open(self.arxiv_pool, "r", encoding="utf-8") as f:
                if arxiv_id in f.read():
                    print(f"{arxiv_id} is already extracted before!")
                    continue
            #add basic info
            print(3)
            add_doc = DocSet(
            doc_id=arxiv_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            categories=result.categories,
            published_date=str(result.published),
            abstract=result.summary,
            pdf_path=str(os.path.join(self.pdf_folder_path, f'{arxiv_id}.pdf')),
            #Set htmlpath to None first and update it later
            HTML_path=None )
            print(4)

            print(arxiv_id)
            #download_arxiv_pdf(arxiv_id, self.pdf_folder_path)
            #result.download_pdf(dirpath = self.pdf_folder_path, filename=f"{arxiv_id}.pdf")
            success = download_paper(
                result=result,
                save_path=self.pdf_folder_path,
                filename=f"{arxiv_id}.pdf"
            )
            if not success:
                print(f"❌ 论文 {result.title} 下载最终失败")

            print(5)

            self.docs.append(add_doc)

        #self.serialize_docs_init()

    def extract_all(self):
        """"All in one function"""
        self.init_docset()
        for doc in self.docs:
            path = doc.pdf_path
            print("getting markdown...")
            markdown_path = get_pdf_md(path,self.pdf_folder_path,doc.doc_id,self.ak,self.sk)
            print("done, begin chunking")
            if markdown_path:
                doc.figure_chunks = self.pdf_images_chunk(markdown_path,self.image_folder_path,doc.doc_id)
                doc.table_chunks = self.pdf_tables_chunk(markdown_path)
                doc.text_chunks = self.pdf_text_chunk(markdown_path)#一定在最后
        self.serialize_docs()

    def remain_docparser(self):
        """
        Help HTMLExtractor. We don't use it in the process of PDF's extractor
        """
        for doc in self.docs:
            if doc.HTML_path is None and doc.pdf_path is not None:
                path = doc.pdf_path
                print("getting markdown...")
                markdown_path = get_pdf_md(path,self.pdf_folder_path,doc.doc_id,self.ak,self.sk)
                print("done, begin chunking")
                if markdown_path:
                    doc.figure_chunks = self.pdf_images_chunk(markdown_path,self.image_folder_path,doc.doc_id)
                    doc.table_chunks = self.pdf_tables_chunk(markdown_path)
                    doc.text_chunks = self.pdf_text_chunk(markdown_path)#一定在最后
            elif doc.pdf_path is None:
                print("Neither PDF or HTML is avaliable.")
                   
    def pdf_images_chunk(self, markdown_path, image_folder_path, doc_id):
        figures = []

        try:
            # read Markdown
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                
            image_list = _parse_image_urls(md_content, doc_id) # Original line, no change based on instruction. The snippet in the prompt was malformed.
            if not image_list:
                print("Warning: No image link was found in Markdown")
                return figures
                
            os.makedirs(image_folder_path, exist_ok=True)
            
            # download
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
            
            print(f"\n📌 Download completed: process{len(image_list)} figures totally, {success_count} Successfully")
            
        except FileNotFoundError:
            print(f"Error: Markdown Not Found - {markdown_path}")
            
        except Exception as e:
            print(f"program exception:{str(e)}")
        
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

                # Find the text position of this table in the Markdown content
                table_pos = md_content.find(table_html)
                context_before = md_content[max(0, table_pos - 500):table_pos]

                # Look for the Table title from the previous text
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
            print(f"Error: Markdown Not Found - {markdown_path}")
        except Exception as e:
            print(f"program exception:{str(e)}")
        #print(tables)
        return tables
    
    def pdf_text_chunk(self, markdown_path) -> List[TextChunk]:
        all_text = []

        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            md_content = re.sub(r'^!\[fig_[^\n]*\n?', '', md_content, flags=re.MULTILINE)

            # 查找所有一级标题（## 开头，排除如 2.1 开头的子标题）
            pattern = r'(?:^|\n)(##\s+(?!(?:[A-Za-z]+\.)?\d+\.\d+)[^\n]+)'
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

############################################################### Some Tools ####################################################################
def compress_pdf(input_path: str, output_path: str = None, max_size_mb: float = 7.5) -> str:
    """
    压缩 PDF 文件，使输出文件大小小于指定的最大 MB。
    若图片重编码与不同质量尝试仍未达到目标，会从最后一页开始逐页裁剪直到符合或裁完。
    
    参数：
        input_path (str): 输入 PDF 路径
        output_path (str): 输出 PDF 路径（如果为 None 则覆盖原文件）
        max_size_mb (float): 最大文件大小（MB）
    
    返回：
        str: 压缩后文件路径（如果覆盖原文件，返回原文件路径）
    """
    # 决定写入路径：如果 output_path 为 None，则写入临时文件，最后替换原文件
    overwrite_original = output_path is None
    if overwrite_original:
        tmp_output = f"{input_path}.compress_tmp"
        write_path = tmp_output
        final_output_path = input_path
    else:
        write_path = output_path
        final_output_path = output_path

    if output_path is None:
        # 保持兼容：不直接覆盖原文件，使用临时文件最后原子替换
        pass
    else:
        # 如果传入的是目录，则在该目录下构造文件名
        if os.path.isdir(output_path):
            base = os.path.splitext(os.path.basename(input_path))[0]
            write_path = os.path.join(output_path, f"{base}_compressed.pdf")
            final_output_path = write_path

    # 输入文件大小（MB）
    try:
        original_size = os.path.getsize(input_path)
    except Exception:
        original_size = None

    if original_size is not None:
        print(f"输入: {input_path} -> {original_size / 1024 / 1024:.2f} MB")
    else:
        print(f"输入: {input_path} -> 无法获取大小")

    max_bytes = max_size_mb * 1024 * 1024
    quality = 90
    step = 10

    # 尝试导入 Pillow，用于图片重编码
    try:
        from PIL import Image
        pillow_available = True
        print("Pillow: 可用，启用图片重编码")
    except Exception:
        pillow_available = False
        print("Pillow: 不可用，仅使用 PyMuPDF 保存参数")

    out_buf = None
    size = None

    # 每次从原文件重新打开，避免重复压缩同一图像
    try:
        while quality >= 10:
            doc = fitz.open(input_path)
            # 若安装了 Pillow，则对文档中所有图片以当前质量重编码
            if pillow_available:
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    images = page.get_images(full=True)
                    if not images:
                        continue
                    for img_info in images:
                        xref = img_info[0]
                        try:
                            img_dict = doc.extract_image(xref)
                            img_bytes = img_dict.get("image")
                            if not img_bytes:
                                continue
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=quality)
                            new_bytes = buf.getvalue()
                            # 用重编码后的字节替换原图
                            doc.update_image(xref, stream=new_bytes)
                        except Exception:
                            # 遇到任何图片处理问题，跳过该图片
                            continue

            # 使用仅受支持的参数保存到内存并检查大小
            out_buf = io.BytesIO()
            doc.save(out_buf, garbage=4, deflate=True, clean=True, incremental=False)
            size = len(out_buf.getvalue())
            doc.close()

            print(f"尝试: 质量={quality} -> {size / 1024 / 1024:.2f} MB")

            if size <= max_bytes:
                # 写入临时或目标文件
                with open(write_path, "wb") as f:
                    f.write(out_buf.getvalue())
                # 如果需要覆盖原文件，原子替换
                if overwrite_original:
                    try:
                        os.replace(write_path, final_output_path)
                    except Exception as e:
                        # 若替换失败，保留临时文件并抛出
                        print(f"⚠️ 无法替换原文件: {e}，临时文件保留在 {write_path}")
                        raise
                if original_size is not None:
                    print(f"✅ 压缩完成: {final_output_path} ({original_size / 1024 / 1024:.2f} MB -> {size / 1024 / 1024:.2f} MB, 质量={quality})")
                else:
                    print(f"✅ 压缩完成: {final_output_path} ({size / 1024 / 1024:.2f} MB, 质量={quality})")
                return final_output_path

            quality -= step

        # 若到这里仍未达标，尝试裁剪最后几页（从最后一页开始裁），每次少一页直到符合或裁完
        print("开始裁剪页面（从最后一页开始逐页裁剪）以尝试达标...")
        try:
            orig_doc = fitz.open(input_path)
        except Exception as e:
            print(f"无法打开原文件进行裁剪: {e}")
            # 若无法打开原文件，直接保存上次生成的 out_buf（若有）到写入路径
            if out_buf is not None:
                with open(write_path, "wb") as f:
                    f.write(out_buf.getvalue())
                if overwrite_original:
                    try:
                        os.replace(write_path, final_output_path)
                    except Exception:
                        pass
            raise

        total_pages = len(orig_doc)
        # 如果文件本来就只有1页且未达标，直接保存当前 out_buf（如果存在）
        if total_pages <= 1:
            print("PDF 只有 1 页，无法裁剪更多页。")
            if out_buf is not None:
                with open(write_path, "wb") as f:
                    f.write(out_buf.getvalue())
                if overwrite_original:
                    try:
                        os.replace(write_path, final_output_path)
                    except Exception:
                        pass
            orig_doc.close()
            if original_size is not None:
                print(f"⚠️ 无法完全压缩到 {max_size_mb} MB 以下，已尽可能压缩 ({original_size / 1024 / 1024:.2f} MB -> {size / 1024 / 1024:.2f} MB)")
            else:
                print(f"⚠️ 无法完全压缩到 {max_size_mb} MB 以下，已尽可能压缩 ({size / 1024 / 1024:.2f} MB)")
            return final_output_path

        # 裁剪循环：保留页数从 total_pages-1 到 1
        success = False
        for keep in range(total_pages - 1, 0, -1):
            new_doc = fitz.open()  # 空文档
            try:
                new_doc.insert_pdf(orig_doc, from_page=0, to_page=keep - 1)
                # 对新文档进行图片重编码（如果 Pillow 可用），使用较低质量以更易达标
                if pillow_available:
                    trim_quality = max(10, quality) if quality >= 10 else 30
                    for page_index in range(len(new_doc)):
                        page = new_doc[page_index]
                        images = page.get_images(full=True)
                        if not images:
                            continue
                        for img_info in images:
                            xref = img_info[0]
                            try:
                                img_dict = new_doc.extract_image(xref)
                                img_bytes = img_dict.get("image")
                                if not img_bytes:
                                    continue
                                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                buf = io.BytesIO()
                                img.save(buf, format="JPEG", quality=trim_quality)
                                new_bytes = buf.getvalue()
                                new_doc.update_image(xref, stream=new_bytes)
                            except Exception:
                                continue

                buf = io.BytesIO()
                new_doc.save(buf, garbage=4, deflate=True, clean=True, incremental=False)
                this_size = len(buf.getvalue())
                print(f"裁剪: 保留 {keep} 页 -> {this_size / 1024 / 1024:.2f} MB")

                if this_size <= max_bytes:
                    with open(write_path, "wb") as f:
                        f.write(buf.getvalue())
                    if overwrite_original:
                        try:
                            os.replace(write_path, final_output_path)
                        except Exception as e:
                            print(f"⚠️ 无法替换原文件: {e}，临时文件保留在 {write_path}")
                            raise
                    if original_size is not None:
                        print(f"✅ 裁剪并压缩完成: {final_output_path} ({original_size / 1024 / 1024:.2f} MB -> {this_size / 1024 / 1024:.2f} MB, 保留 {keep} 页)")
                    else:
                        print(f"✅ 裁剪并压缩完成: {final_output_path} ({this_size / 1024 / 1024:.2f} MB, 保留 {keep} 页)")
                    success = True
                    new_doc.close()
                    break
            finally:
                try:
                    new_doc.close()
                except Exception:
                    pass

        orig_doc.close()

        if not success:
            # 无法通过裁剪达到目标，保存最后一次尝试的结果（尽可能压缩）
            if out_buf is not None:
                with open(write_path, "wb") as f:
                    f.write(out_buf.getvalue())
                if overwrite_original:
                    try:
                        os.replace(write_path, final_output_path)
                    except Exception:
                        pass
            if original_size is not None and size is not None:
                print(f"⚠️ 裁剪后仍无法完全压缩到 {max_size_mb} MB 以下，已尽可能压缩 ({original_size / 1024 / 1024:.2f} MB -> {size / 1024 / 1024:.2f} MB)")
            elif size is not None:
                print(f"⚠️ 裁剪后仍无法完全压缩到 {max_size_mb} MB 以下，已尽可能压缩 ({size / 1024 / 1024:.2f} MB)")
        return final_output_path
    finally:
        # 清理可能残留的临时文件（仅当写入临时覆盖原文件且最终文件已被替换时）
        if overwrite_original:
            tmp = f"{input_path}.compress_tmp"
            if os.path.exists(tmp):
                # 如果原子替换已经发生，临时文件可能已被移动；仅在还存在时尝试删除
                try:
                    os.remove(tmp)
                except Exception:
                    pass

def verify_pdf(file_path: str) -> bool:
    """
    验证PDF文件是否有效
    
    Args:
        file_path: PDF文件路径
    
    Returns:
        bool: 文件是否有效
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                print(f"❌ {file_path} 不是有效的PDF文件")
                return False
        return True
    except Exception as e:
        print(f"❌ 验证PDF文件失败 {file_path}: {str(e)}")
        return False

def download_pdf_with_retry(url: str, save_path: str, filename: str, max_retries: int = 3) -> bool:
    """
    使用重试机制下载PDF文件
    
    Args:
        url: PDF文件的URL
        save_path: 保存路径
        filename: 文件名
        max_retries: 最大重试次数
    
    Returns:
        bool: 下载是否成功
    """
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; AIgniteBot/1.0; +https://github.com/Algnite-Solutions/AIgnite)',
        'Accept': 'application/pdf'
    }
    
    temp_path = os.path.join(save_path, f"{filename}.tmp")
    final_path = os.path.join(save_path, filename)
    
    try:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 下载文件
        response = session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 使用临时文件下载
        with open(temp_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        # 验证文件大小
        if total_size > 0 and downloaded_size != total_size:
            raise ValueError(f"文件大小不匹配: 预期 {total_size} 字节，实际下载 {downloaded_size} 字节")
        
        # 验证PDF文件
        with open(temp_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                raise ValueError("文件不是有效的PDF格式")
        
        # 如果验证通过，重命名临时文件
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)
        
        print(f"✅ 成功下载: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败 {filename}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def download_paper(result, save_path: str, filename: str) -> bool:
    """
    下载论文，先尝试使用arxiv API，如果失败则使用可靠下载方法
    
    Args:
        result: arxiv搜索结果
        save_path: 保存路径
        filename: 文件名
    
    Returns:
        bool: 下载是否成功
    """
    file_path = os.path.join(save_path, filename)
    
    # 第一步：尝试使用arxiv API下载
    try:
        print(f"尝试使用arxiv API下载: {filename}")
        result.download_pdf(dirpath=save_path, filename=filename)
        
        # 验证下载的文件
        if verify_pdf(file_path):
            print(f"✅ arxiv API下载成功: {filename}")
            return True
        else:
            print("⚠️ arxiv API下载的文件无效，尝试使用可靠下载方法...这可能需要稍微长一点的时间。")
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"❌ arxiv API下载失败: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # 第二步：使用可靠下载方法
    print(f"使用可靠下载方法下载: {filename}")
    return download_pdf_with_retry(
        url=result.entry_id.replace("abs", "pdf"),
        save_path=save_path,
        filename=filename
    )

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

def get_pdf_md(path,store_path,name,ak,sk):
    visual_service = VisualService()
    # call below method if you dont set ak and sk in $HOME/.volc/config
    visual_service.set_ak(ak)
    visual_service.set_sk(sk)

    # params = dict()  # Unused variable
    pdf_content = None

    # 使用 with 语句确保文件正确关闭
    with open(str(path), 'rb') as f:
        pdf_content = f.read()
        
    if os.path.getsize(path) > 7.5*1024*1024:
        print("📦 PDF 超过 7.5MB，需要压缩。")
        try:
            compressed_path = compress_pdf(path)
            # compress_pdf在失败时会返回原文件路径，检查是否真的压缩了
            original_size = os.path.getsize(path)
            compressed_size = os.path.getsize(compressed_path) if os.path.exists(compressed_path) else original_size
            
            if compressed_size < original_size:
                print(f"✅ PDF压缩完成，使用压缩后的文件 ({compressed_size/(1024*1024):.2f}MB)")
                path = compressed_path
            else:
                print(f"⚠️ PDF压缩未生效，使用原文件 ({original_size/(1024*1024):.2f}MB)")
            
            with open(str(path), 'rb') as f:
                pdf_content = f.read()
        except Exception as e:
            print(f"⚠️ 压缩过程出错：{e}，尝试使用原文件")
            # 即使压缩失败，也尝试使用原文件继续处理
            try:
                with open(str(path), 'rb') as f:
                    pdf_content = f.read()
            except Exception as e2:
                print(f"❌ 无法读取PDF文件：{e2}")
                return None
    
    form = {
        "image_base64": base64.b64encode(pdf_content).decode(),   # 文件binary 图片/PDF 
        "image_url": "",                  # url
        "version": "v3",                  # 版本
        "page_start": 0,                  # 起始页数
        "page_num": 16,                   # 解析页数
        "table_mode": "html",             # 表格解析模式
        "filter_header": "true"           # 过滤页眉页脚水印
    }

    # 请求
    try:
        resp = visual_service.ocr_pdf(form)
        
        # 增强错误处理
        if resp is None:
            print("❌ OCR请求失败：服务返回空响应")
            return None
            
        if not isinstance(resp, dict):
            print(f"❌ OCR请求失败：响应类型错误，期望dict，实际{type(resp)}")
            return None
            
        if "data" not in resp:
            print(f"❌ OCR请求失败：响应中缺少'data'字段，实际字段: {list(resp.keys())}")
            return None
            
        if resp["data"] is None:
            print("❌ OCR请求失败：data字段为None")
            return None
            
        markdown = resp["data"].get("markdown")
        if not markdown:
            print("❌ OCR请求失败：未获取到markdown内容")
            print(f"响应data内容: {resp['data']}")
            return None

        # 确保目录存在
        os.makedirs(store_path, exist_ok=True)
        # 完整文件路径
        file_path = os.path.join(store_path, f"{name}.md")

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return file_path

    except Exception as e:
        print(f"❌ OCR请求失败：{str(e)}")
        return None

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
        
        # 首字母大写
        cleaned_name = cleaned_name[0].upper() + cleaned_name[1:]
        
        # 将Fig开头转换为Figure开头
        if cleaned_name.startswith('Fig') and not cleaned_name.startswith('Figure'):
            cleaned_name = 'Figure' + cleaned_name[3:]
            
        return cleaned_name
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
        '''ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            ext = 'png'  # 默认扩展名'''
        
        ext = 'png'
        
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