from abc import ABC, abstractmethod
import asyncio
import aiohttp
from typing import List
import os
from google import genai
from google.genai import types
from AIgnite.data.docset import DocSet
from tqdm import tqdm


class BaseGenerator(ABC):
    """
    Abstract base class for LLM generators.
    """
    @abstractmethod
    def generate_response(self):
        pass


class GeminiGenerator(BaseGenerator):
    """
    A simple Gemini generator that takes a text prompt and returns text output.
    """
    def __init__(self, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key="AIzaSyDQS4jFfedzDourgwQxiP4hhOR0lK67l44")
        self.model_name = model_name

    def generate_response(self, prompt: str, pdf_path: str = None) -> str:
        """
        Generate a text response from a text prompt, optionally with PDF input.
        
        Args:
            prompt (str): The input text prompt
            pdf_path (str, optional): Path to PDF file to include in the request
            
        Returns:
            str: The generated text response
        """
        contents = []
        
        if pdf_path:
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            contents.append(
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                )
            )
        
        contents.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        return response.text

class GeminiBlogGenerator(BaseGenerator):
    """
    A class to generate blog posts using the Gemini model.
    This class uses the Google Gemini model to generate blog posts based on the provided PDF documents.
    TODO: @Qi, replace data_path and output_path with the actual DB_query and DB_write functions.
    """
    def __init__(self, model_name="gemini-2.5-flash", data_path="./output", output_path="./experiments/output"):
        self.client = genai.Client(api_key="AIzaSyDQS4jFfedzDourgwQxiP4hhOR0lK67l44")
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def generate_response(self, papers: List[DocSet]):
        for paper in papers:
            self._generate_single_blog(paper)

    def _generate_single_blog(self, paper: DocSet):
        # Read and encode the PDF bytes
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        arxiv_id = paper.doc_id

        prompt = f"""
        You're generating a mark down blog post summarizing a paper with arXiv ID {arxiv_id} for researchers in the field. The style is similar to medium science blog.

        In your blog, you can cite a **few of the most important figures** from the paper (ideally no more than 3) to help understanding. For each selected figure, render it as a standalone Markdown image:
          <br>![Figure X: short caption]({self.data_path}/{arxiv_id}_FigureX.png)<br>

        Do **not** use inline figure references like “as shown in Figure 2”. Do **not** cite tables.
        Start directly with Blog title.
        """

        prompt = f"""
        你是一个专业的科技博客作者，专门为中国的研究人员撰写学术论文的中文博客总结。
      你的任务是：
      1. 突出论文的核心贡献和创新点
      2. 使用 Medium 科技博客的写作风格
      3. 引用重要的图表来帮助理解（最多3个）
      4. 直接以博客标题开始，不要添加任何前缀

      我将给你一篇论文的详细内容，请为以下论文生成一篇博客文章。
      
      请确保博客内容：
      - 结构清晰，逻辑连贯，尽量详细一些，不要过于简略
      - 在博客前几部分突出论文的核心贡献，符合新闻学博人眼球的风格
      - 重点介绍文章的比较重要的方法，并且引用pipeline图，并且给出pipeline图的解释
      - 适合研究人员阅读，但不要晦涩难懂，在必要的地方可以适当解释复杂的名词概念

      请使用小标题。你最好可以根据文章实际内容确定一些针对本篇文章特有的小标题。不要设置层次过多的小标题。
      最好可以在开头有吸引人的小标题
      最好小标题具有强大的概括能力，显得很精辟
      如果你实在没有灵感的话，你可以参考的小标题：
      - 概述介绍
      - 理论框架和定义
      - 核心方法
      - 实验设计
      - 应用场景及评估
      - 未来发展方向和开放性挑战
      - 相关引文
      - 相关链接

      注意事项：
      如果论文包含图表，请选择重要的图表（尤其是表示pipeline的图）进行引用。
      对于每个图表，使用以下格式：
      ![Figure X: short caption]({self.data_path}/{arxiv_id}_FigureX.png)

      论文的额外信息（如官方网站、代码、数据集等）可以使用超链接。

      论文标题：{paper.title}
      作者：{paper.authors}
      摘要：{paper.abstract}
      论文内容：{paper.text_chunks}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )

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