from abc import ABC, abstractmethod
from typing import List
import os
from google import genai
from google.genai import types
from AIgnite.data.docset import DocSet
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import re
import json
'''

pip install einops transformers_stream_generator

TODO: Use Qwen3

'''
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
        
        Please Cite Figure!!!!!!!
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

class Qwen72BAgent(BaseGenerator):
    def __init__(self, model_path="Qwen/Qwen-72B-Chat", device=None, data_path="./output", output_path="./experiments/output"):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            do_sample=True
        )
        self.data_path = data_path
        self.output_path = output_path

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def generate_digest(self, papers: List[DocSet]):
        """Generate blog digests for a list of papers."""
        for paper in papers:
            self._generate_single_blog(paper)

    def _generate_single_blog(self, paper: DocSet):
        """Generate a single blog post for a paper."""
        arxiv_id = paper.doc_id

        # Create prompt similar to GeminiBlogGenerator
        prompt = f"""
        You're generating a mark down blog post summarizing a paper with arXiv ID {arxiv_id} for researchers in the field. The style is similar to medium science blog.

        In your blog, you must cite a **few of the most important figures** from the paper (ideally no more than 3) to help understanding. For each selected figure, render it as a standalone Markdown image:
          <br>![Figure X: short caption]({self.data_path}/{arxiv_id}_FigureX.png)<br>

        Do **not** use inline figure references like "as shown in Figure 2". Do **not** cite tables.
        Start directly with Blog title.
        """

        # Generate response using Qwen model
        response = self.generate_response(prompt)

        print("===RAW RESPONSE===")
        print(response)
        print("===END RAW RESPONSE===")

        if response.strip().startswith(prompt.strip()):
            cleaned = response[len(prompt):].lstrip()
        else:
            cleaned = response

        # Save the generated blog
        markdown_path = os.path.join(self.output_path, f"{arxiv_id}.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(cleaned)

        print(f"✅ Markdown file saved to {markdown_path}")

    def run_agent_task(self, task_description):
        system_prompt = "You are a helpful AI agent designed to complete tasks accurately."
        full_prompt = f"<system>{system_prompt}</system><user>{task_description}</user>"
        return self.generate_response(full_prompt)

if __name__ == "__main__":
    agent = Qwen72BAgent(
        data_path="../imgs",
        output_path="/data3/guofang/peirongcan/generator-dev/blogs"
    )

    # 从 JSON 文件读取 DocSet 数据
    dummy_json_path = "/data3/guofang/peirongcan/AIgnite/src/AIgnite/dummy_data/dummy.json"
    with open(dummy_json_path, "r", encoding="utf-8") as f:
        docset_data = json.load(f)

    # 如果 dummy.json 是单个 DocSet
    dummy_paper = DocSet.parse_obj(docset_data)
    agent._generate_single_blog(dummy_paper)

    # 如果 dummy.json 是 DocSet 列表
    # papers = [DocSet.parse_obj(item) for item in docset_data]
    # agent.generate_digest(papers)