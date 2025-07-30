from abc import ABC, abstractmethod
import asyncio
import aiohttp
from typing import List
import os
from google import genai
from google.genai import types
from AIgnite.data.docset import DocSet


class BaseGenerator(ABC):
    """
    Abstract base class for blog generators.
    """
    @abstractmethod
    def generate_digest(self):
        pass


class GeminiBlogGenerator(BaseGenerator):
    """
    A class to generate blog posts using the Gemini model.
    This class uses the Google Gemini model to generate blog posts based on the provided PDF documents.
    TODO: @Qi, replace data_path and output_path with the actual DB_query and DB_write functions.
    """
    def __init__(self, model_name="gemini-2.5-flash-preview-04-17", data_path="./output", output_path="./experiments/output"):
        self.client = genai.Client()
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def generate_digest(self, papers: List[DocSet]):
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
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)

        print(f"✅ Markdown file saved to {markdown_path}")
        print("📊 Token usage:", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)


class AsyncvLLMGenerator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", api_base="http://localhost:8000/v1"):
        self.model_name = model_name
        self.api_base = api_base

    async def generate_response(self, session, prompt, system_prompt, max_tokens=2048):
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
            "max_tokens": max_tokens,
        }
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Error {resp.status}: {text}")
            response = await resp.json()
            # Return the generated text.
            return response["choices"][0]["message"]["content"]
    async def batch_generate(self, prompts, system_prompts=None, max_tokens=2048):
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
                task = asyncio.create_task(self.generate_response(session, prompt, sys_prompt, max_tokens))
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