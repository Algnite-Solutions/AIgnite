from bs4 import BeautifulSoup
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
            print(f"⚠️ arxiv API下载的文件无效，尝试使用可靠下载方法...这可能需要稍微长一点的时间。")
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"❌ arxiv API下载失败: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # 第二步：使用可靠下载方法
    print(f"使用可靠下载方法下载: {filename}")
    return download_pdf_with_retry(
        url=result.pdf_url,
        save_path=save_path,
        filename=filename
    )

date = datetime.now(timezone.utc).date()

client = arxiv.Client()
one_day = timedelta(days=1)
yesterday = date - one_day

exact_time = "0600"
today_str = date.strftime("%Y%m%d") + exact_time
yesterday_str = yesterday.strftime("%Y%m%d") + exact_time
print(today_str)
query = "cat:cs.* AND submittedDate:[" + yesterday_str + " TO " + today_str + "]"
#query = "cat:cs.* AND submittedDate:[202504190900 TO 202504191200]"
print(today_str,yesterday_str)

search = arxiv.Search(
    query=query,
    max_results=8,  # You can set max papers you want here
    sort_by=arxiv.SortCriterion.SubmittedDate
)

i = 0
for r in client.results(search):
    i = i+1
    print(f"正在下载第 {i} 篇论文: {r.title}")
    
    # 使用改进的下载函数
    success = download_paper(
        result=r,
        save_path="/data3/peirongcan/paperIgnite/AIgnite/test/pdfs",
        filename=f"{i}.pdf"
    )
    
    if not success:
        print(f"❌ 论文 {r.title} 下载最终失败")