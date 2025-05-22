import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 日志配置（可选）
logging.basicConfig(level=logging.INFO)

class SafeSession(requests.Session):
    def __init__(self, retries=3, backoff=1, timeout=15):
        super().__init__()
        retry_cfg = Retry(
            total=retries,
            connect=retries,
            read=retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.mount("http://", adapter)
        self.mount("https://", adapter)
        self.timeout = timeout
        self.headers["User-Agent"] = (
            "AIgniteBot/1.0 (+https://github.com/AIgnite-Solutions)"
        )

    def safe_get(self, url: str, **kwargs) -> requests.Response | None:
        try:
            resp = super().get(url, timeout=kwargs.pop("timeout", self.timeout), **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as ex:
            logging.warning("GET %s 失败：%s", url, ex)
            return None

def download_image_to_file(url: str, save_path: str, session: SafeSession | None = None):
    """下载图片到本地PNG文件。如果失败，写入空文件"""
    session = session or SafeSession()
    response = session.safe_get(url)
    
    # 确保目标路径目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if response and response.headers.get("Content-Type", "").startswith("image"):
        try:
            with open(save_path, "wb") as f:
                f.write(response.content)
            logging.info("✅ 下载成功: %s", save_path)
        except Exception as e:
            logging.error("写入文件失败 %s: %s", save_path, e)
            with open(save_path, "wb") as f:
                pass  # 空文件占位
    else:
        logging.warning("❌ 无法下载图片或响应不是图片: %s", url)
        with open(save_path, "wb") as f:
            pass  # 写空文件占位

# 示例用法
if __name__ == "__main__":
    img_url = "https://arxiv.org/html/2505.10876v1/x8.png"
    save_path = "/data3/peirongcan/paperIgnite/AIgnite/src/AIgnite/data/tem.png"
    download_image_to_file(img_url, save_path)
