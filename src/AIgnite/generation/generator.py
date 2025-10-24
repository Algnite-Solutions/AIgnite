from abc import ABC, abstractmethod
import asyncio
import aiohttp
from typing import List
import os
from google import genai
from google.genai import types
from AIgnite.data.docset import DocSet
from tqdm import tqdm
import time
from .prompt_config import format_blog_prompt

class BaseGenerator(ABC):
    """
    Abstract base class for blog generators.
    """
    @abstractmethod
    def generate_digest(self):
        pass


class GeminiBlogGenerator_default(BaseGenerator):
    """
    A class to generate blog posts using the Gemini model.
    This class uses the Google Gemini model to generate blog posts based on the provided PDF documents.
    TODO: @Qi, replace data_path and output_path with the actual DB_query and DB_write functions.
    """
    def __init__(self, model_name="gemini-2.5-flash-lite-preview-09-2025", data_path="./output", output_path="./experiments/output"):
        self.client = genai.Client(api_key=your api key)
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def generate_digest(self, papers: List[DocSet]):
        import concurrent.futures
        import threading
        
        def generate_with_delay(paper):
            self._generate_single_blog(paper)
            time.sleep(5)
        
        # 使用线程池进行并行处理，限制最大并发数避免API限制
        max_workers = min(len(papers), 50)  # 最多50个并发任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_with_delay, paper) for paper in papers]
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理论文时出错: {e}")

    def _generate_single_blog(self, paper: DocSet):
        # Read and encode the PDF bytes
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        arxiv_id = paper.doc_id
        prompt = format_blog_prompt(
            data_path=self.data_path,
            arxiv_id=arxiv_id,
            text_chunks=paper.text_chunks,
            table_chunks=paper.table_chunks,
            figure_chunks=paper.figure_chunks
        )
        import time

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        prompt,
                        ''' types.Part.from_bytes(
                            data=pdf_data,
                            mime_type='application/pdf',
                        ),'''
                    ]
                )
                break  # 成功就跳出循环
            except Exception as e:
                print(f"处理论文时出错（第 {attempt} 次尝试）: {e}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt  # 指数退避（2, 4, 8, 16...秒）
                    print(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print("已达到最大重试次数，终止。")
                    return

        """
        types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
        """

        markdown_path = os.path.join(self.output_path, f"{arxiv_id}.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)  # 确保目录存在
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)

        print(f"✅ Markdown file saved to {markdown_path}")
        print("📊 Token usage:", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

class GeminiBlogGenerator_recommend(BaseGenerator):
    """
    A class to generate blog posts using the Gemini model.
    This class uses the Google Gemini model to generate blog posts based on the provided PDF documents.
    TODO: @Qi, replace data_path and output_path with the actual DB_query and DB_write functions.
    """
    def __init__(self, model_name="gemini-2.5-flash-preview-09-2025", data_path="./output", output_path="./experiments/output"):
        self.client = genai.Client(api_key=your api key)
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def generate_digest(self, papers: List[DocSet]):
        import concurrent.futures
        import threading
        
        def generate_with_delay(paper):
            self._generate_single_blog(paper)
            time.sleep(5)
        
        # 使用线程池进行并行处理，限制最大并发数避免API限制
        max_workers = min(len(papers), 50)  # 最多50个并发任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_with_delay, paper) for paper in papers]
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理论文时出错: {e}")

    def _generate_single_blog(self, paper: DocSet):
        # Read and encode the PDF bytes
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        arxiv_id = paper.doc_id
        prompt = format_blog_prompt(
            data_path=self.data_path,
            arxiv_id=arxiv_id,
            text_chunks=paper.text_chunks,
            table_chunks=paper.table_chunks,
            figure_chunks=paper.figure_chunks
        )
        import time

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        prompt,
                        ''' types.Part.from_bytes(
                            data=pdf_data,
                            mime_type='application/pdf',
                        ),'''
                    ]
                )
                break  # 成功就跳出循环
            except Exception as e:
                print(f"处理论文时出错（第 {attempt} 次尝试）: {e}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt  # 指数退避（2, 4, 8, 16...秒）
                    print(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print("已达到最大重试次数，终止。")
                    return

        """
        types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
        """

        markdown_path = os.path.join(self.output_path, f"{arxiv_id}.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)  # 确保目录存在
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)

        print(f"✅ Markdown file saved to {markdown_path}")
        print("📊 Token usage:", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

class AsyncvLLMGenerator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", api_base="http://localhost:8000/v1",data_path="./output", output_path="./experiments/output"):
        self.model_name = model_name
        self.api_base = api_base
        self.data_path = data_path
        self.output_path = output_path

    async def generate_response(self, session, prompt, system_prompt, max_tokens=2048, arxiv_id=None):
        """
        Send a single chat completion request to the vLLM server.
        """
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            #"max_tokens": max_tokens,
        }
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Error {resp.status}: {text}")
            response = await resp.json()

            if arxiv_id is None:
                arxiv_id = 111
            markdown_path = os.path.join(self.output_path, f"{arxiv_id}.md")
            os.makedirs(os.path.dirname(markdown_path), exist_ok=True)  # 确保目录存在
            with open(markdown_path, "w", encoding="utf-8") as md_file:
                content = response["choices"][0]["message"]["content"]
                #在这里增加逻辑，匹配<think>和</think>之间的内容，去掉它，将剩余部分保存到markdown_path中
                think_start = content.find("<think>")
                think_end = content.find("</think>")
                if think_start != -1 and think_end != -1:
                    think_content = content[think_start:think_end + 8]
                    content = content.replace(think_content, "")
                md_file.write(content)
            print(f"✅ Markdown file saved to {markdown_path}")
            print("📊 Token usage:", response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"])
            # Return the generated text.
            return response["choices"][0]["message"]["content"]
        
    async def batch_generate(self, prompts, system_prompts=None, max_tokens=2048, papers: List[DocSet] = None):
        """
        Batch generate responses concurrently using asynchronous HTTP requests,
        preserving the order of prompts and updating progress.

        Args:
            prompts (list): List of input prompts.
            system_prompts (str or list, optional): A single system prompt or list of prompts.
            max_tokens (int, optional): Maximum number of tokens to generate.

        Returns:
            list: Generated responses in the same order as the input prompts.
        """
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif system_prompts is None:
            system_prompts = [''] * len(prompts)
        if len(system_prompts) != len(prompts):
            raise ValueError("Length of system_prompts must match the length of prompts")

        timeout = aiohttp.ClientTimeout(total=None)  # No hardcoded timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            pbar = tqdm(total=len(prompts), desc="Processing")
            
            # Create indexed tasks to track the original order
            for i, (prompt, sys_prompt) in enumerate(zip(prompts, system_prompts)):
                task = asyncio.create_task(self.generate_response(session, prompt, sys_prompt, max_tokens, papers[i].doc_id))
                task.add_done_callback(lambda fut: pbar.update(1))
                tasks.append((i, task))
            
            # Wait for all tasks to complete
            done, _ = await asyncio.wait([t[1] for t in tasks])
            
            # Retrieve results in the original order
            responses = [None] * len(prompts)
            for i, task in tasks:
                responses[i] = task.result()

            pbar.close()
            return responses

    def generate_digest(self, papers: List[DocSet]):
        return 0