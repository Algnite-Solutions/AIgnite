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

class Qwen3_32B(BaseGenerator):
    def __init__(self, model_name = "Qwen/Qwen3-32B", data_path="./output", output_path="./experiments/output"):

        self.data_path = data_path
        self.output_path = output_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_digest(self, papers: List[DocSet]):
        """实现抽象方法 - 为多篇论文生成博客摘要"""
        for paper in papers:
            self._generate_single_blog(paper)

    def _generate_single_blog(self, paper: DocSet):
        """为单篇论文生成博客"""
        arxiv_id = paper.doc_id
        text = paper.text_chunks

        img_path = f"{self.data_path}/{arxiv_id}_FigureX.png"
        prompt = f"""
        您正在基于原始论文生成一篇带有 arXiv 编号 {arxiv_id} 的 Markdown 格式中文博客文章，旨在为该领域的研究人员提供该论文的概要。风格类似于媒体科学博客。

        在您的博客中，您必须引用论文中的**几个最重要的图表**来帮助理解，尤其是概述工作框架的图。对于每个选定的图表，将其渲染为独立的 Markdown 图片：
            ![图表 X: 简短说明]({img_path})

        请勿使用类似“如图2所示”的行内图表引用。请勿引用表格。
        直接从博客标题开始。

        请使用适当的小标题对文章内容进行分隔。包括但不限于：
        背景介绍
        问题背景
        工作贡献
        方法
        实验
        结论
        参考文献
        等等···

        原始论文如下：
        {text}
        """
        
        content = self.speak(prompt)
        
        # 保存生成的博客
        markdown_path = os.path.join(self.output_path, f"{arxiv_id}.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(content)
        
        print(f"✅ Markdown file saved to {markdown_path}")

    def speak(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parse thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        #thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        #print("thinking content:", thinking_content)
        #print("content:", content)
        return content


if __name__ == "__main__":
    # 从 JSON 文件读取 DocSet 数据
    dummy_json_path = "/data3/guofang/peirongcan/AIgnite/src/AIgnite/dummy_data/dummy.json"
    with open(dummy_json_path, "r", encoding="utf-8") as f:
        docset_data = json.load(f)
    # 如果 dummy.json 是单个 DocSet
    dummy_paper = DocSet.parse_obj(docset_data)
    
    agent = Qwen3_32B(
        data_path="/data3/guofang/peirongcan/PaperIgnition/orchestrator/imgs",
        output_path="/data3/guofang/peirongcan/AIgnite/src/AIgnite/dummy_data"
    )
    
    agent._generate_single_blog(dummy_paper)