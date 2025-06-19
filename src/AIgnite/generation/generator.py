from abc import ABC, abstractmethod
from typing import List
import os
from google import genai
from google.genai import types
from AIgnite.data.docset import DocSet
import re

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
    def __init__(self, model_name="gemini-2.5-flash-preview-04-17", data_path="./output", output_path="./experiments/output", api_key = ""):
        self.client = genai.Client(api_key = api_key)
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def generate_digest(self, papers: List[DocSet]):
        for paper in papers:
            path = self._generate_single_blog(paper)
            self.md_clean(path)

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
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)

        print(f"✅ Markdown file saved to {markdown_path}")
        print("📊 Token usage:", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

        return markdown_path

    def md_clean(self, path):
        """
        清洗掉非法路径
        """
        with open(path, "rb") as md_file:
            md_data = md_file.read()

        # 按行分割文本
        lines = md_data.decode('utf-8', errors='ignore').split('\n')
        
        # 收集所有图片路径
        img_paths = []
        for line in lines:
            if '<br>' in line and '../imgs//' in line and '.png' in line:
                start_idx = line.find('../imgs//')
                end_idx = line.find('.png') + 4  # +4 是为了包含.png
                if start_idx != -1 and end_idx != -1:
                    img_path = line[start_idx:end_idx]
                    img_paths.append(img_path)

        # 找到本地不存在的图片路径
        filtered_paths = []
        for img_path in img_paths:
            local_path = os.path.join(os.path.dirname(path), img_path)
            if not os.path.exists(local_path):
                filtered_paths.append(img_path)

        print("图片引用：",img_paths)
        print("不合规引用：",filtered_paths)

        if filtered_paths != []:
            print(f"文档{path}正在清洗。")
            prompt = f"""
            这是一份存在问题的markdown文件，需要你帮忙修正。
            这份md文件一共引用了{len(img_paths)}张图片，分别是{str(img_paths)}，但是在其中，{str(filtered_paths)}这些路径是不存在的。
            请你删掉文件中{filtered_paths}这些图片路径的引用以及对图片的文本描述，然后再整合、润色一下内容。
            注意不要修改任何markdown的格式。
            注意不要删掉别的图片，只删掉这些引用了不合规路径的图片。
            你生成的必须是纯markdown文件，和原来的markdown文件相比，文章结构、字数保持近似，只是掩去了几个图片的引用而已。
            不要添加额外的某些特殊标记比如```markdown等等。
            """

            # 调用API生成内容并保存
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=md_data, mime_type='text/markdown'),
                    prompt
                ]
            )
            
            # 保存为Markdown文件
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as md_file:
                md_file.write(response.text)


class PopularScienceBlogGenerator(GeminiBlogGenerator):
    """生成科普风格的中文博客"""
    def _generate_single_blog(self, paper: DocSet):
        # 读取PDF
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        arxiv_id = paper.doc_id
        
        # 调整prompt以生成科普风格的中文博客
        prompt = f"""
        你是一位知名的科普作家，擅长用较为形象但是不失专业性的语言向大众解释复杂的科学概念。
        请为arXiv ID为{arxiv_id}的学术论文撰写一篇中文科普博客，面向对该领域有兴趣的有一定专业知识的读者。

        博客基本要求：
        博客**必须**以纯markdown的形式出现，请你不要返回别的文本。注意适当使用#建立标题，使用##，###来建立小标题。
        博客**必须**直接以#博客标题开始撰写。
        博客**必须**按图表引用规则引用论文最主要的最重要的至少一张图片（文章不包含图片或不包含主要框架图片除外），但也要注意详略得当，不要引用一些无关紧要的图片。
        博客**尽量**开头铺垫不要太多，尽量简单直接地进入主题，顺带介绍论文是哪个机构的工作。
        博客**尽量**注意详略得当，文章花大篇幅介绍的地方你也要着重介绍。
        
        **必须遵守**图表引用规则：
        引用论文中的重要图表时，使用以下格式（注意千万不要出现不存在的Figure序号）：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X和Y替换为实际上的数字，其余的不可变动。

        **注意**：
        图片的命名是严格格式化的，一定是：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X代表数字或数字+括号数字，比如Figure2代表图2，Figure2(1)代表图2中的第一个子图，可能文中的描述是Figure2(a)，但不管怎样你在调用时需要将其转化为数字，比如第x个子图的索引就是Figure2(x)。
        注意F要大写。

        **必须不能**引用文中不存在的图片
        **必须不能**图片引用写成了Algorithm或http开头的网址等，比如{self.data_path}/{arxiv_id}_AlgorithmX.png。
        
        **示例**
        一些正确的图片引用示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(1).png)<br>
        错误图片命名示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Algorithm2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(a).png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_title2.png)<br>
        <br>![图2：简短说明](https://github.com/jeff-zhuo/paper_figure_captions/blob/main/{self.data_path}/{arxiv_id}_Figure5.png)<br>

        博客风格要求：
        语言生动有趣，但不要忘记使用专业的术语。
        适当添加比喻、类比等修辞手法，这个不要强求，如果没有合适的修辞手法就正常平铺直叙。
        保持内容的科学性和准确性。
        
        """
        
        # 调用API生成内容并保存
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                prompt
            ]
        )
        
        # 保存为Markdown文件
        markdown_path = os.path.join(self.output_path, f"{arxiv_id}_科普.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)
        
        return markdown_path

class AcademicBlogGenerator(GeminiBlogGenerator):
    """生成学术风格的中文博客"""
    def _generate_single_blog(self, paper: DocSet):
        # 针对学术同行的深度解析
        # 读取PDF
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        arxiv_id = paper.doc_id
        
        # 调整prompt以生成科普风格的中文博客
        prompt = f"""
        你是该论文领域的资深研究员，您正在为该领域的研究人员生成一篇带有arXiv ID {arxiv_id}的论文总结的标记中文博客文章。
        请为arXiv ID为{arxiv_id}的学术论文撰写一篇中文科普博客，面向对该领域的科研人员。

        博客基本要求：
        博客**必须**以纯markdown的形式出现，请你不要返回别的文本。注意适当使用#建立标题，使用##，###来建立小标题。
        博客**必须**直接以#博客标题开始撰写。
        博客**必须**按图表引用规则引用论文最主要、最重要的至少一张图片（文章不包含图片或不包含主要框架图片除外），但也要注意详略得当，不要引用一些无关紧要的图片。
        博客**尽量**开头铺垫不要太多，尽量简单直接地进入主题，顺带介绍论文是哪个机构的工作。
        博客**尽量**注意详略得当，文章花大篇幅介绍的地方你也要着重介绍。
        
        **必须遵守**图表引用规则：
        引用论文中的重要图表时，使用以下格式（注意千万不要出现不存在的Figure序号）：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X和Y替换为实际上的数字，其余的不可变动。

        **注意**：
        图片的命名是严格格式化的，一定是：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X代表数字或数字+括号数字，比如Figure2代表图2，Figure2(1)代表图2中的第一个子图，可能文中的描述是Figure2(a)，但不管怎样你在调用时需要将其转化为数字，比如第x个子图的索引就是Figure2(x)。
        注意F要大写。

        **必须不能**引用文中不存在的图片
        **必须不能**图片引用写成了Algorithm或http开头的网址等，比如{self.data_path}/{arxiv_id}_AlgorithmX.png。
        
        **示例**
        一些正确的图片引用示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(1).png)<br>
        错误图片命名示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Algorithm2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(a).png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_title2.png)<br>
        <br>![图2：简短说明](https://github.com/jeff-zhuo/paper_figure_captions/blob/main/{self.data_path}/{arxiv_id}_Figure5.png)<br>

        博客风格要求：
        语言严谨，准确传达论文的技术细节
        分析方法的创新点和实验结果
        讨论该研究的理论意义和实际应用价值
        适当比较相关工作的优缺点
        """
        # 调用API生成内容并保存
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                prompt
            ]
        )
        
        # 保存为Markdown文件
        markdown_path = os.path.join(self.output_path, f"{arxiv_id}_专业.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)

class QABlogGenerator(GeminiBlogGenerator):
    """生成问答风格的中文博客"""
    def _generate_single_blog(self, paper: DocSet):
        # 采用问答形式解析论文
        # 读取PDF
        with open(paper.pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        arxiv_id = paper.doc_id
        
        # 调整prompt以生成科普风格的中文博客
        prompt = f"""
        您正在为该领域的研究人员生成一篇带有arXiv ID {arxiv_id}的论文总结的标记中文博客文章。
        请以问答形式为arXiv ID为{arxiv_id}的论文撰写一篇中文博客。
        博客以纯markdown的形式出现，请你不要返回别的文本，必须是纯markdown格式。
        假设读者已经具备该领域的基础知识，但不熟悉这篇具体的论文。
        
        博客结构要求：
        1. 这篇论文是什么机构发表的，解决了什么问题？创新点是什么？
        2. 作者主要方法是什么？请在这一部分引用文章中最宏观的图片。
        3. 实验结果如何？有哪些关键发现？
        4. 这项研究有什么局限性？未来的研究方向是什么？
        每个问题用清晰的标题标出，如"一、这篇论文解决了什么问题？"

        博客基本要求：
        博客**必须**以纯markdown的形式出现，请你不要返回别的文本。注意适当使用#建立标题，使用##，###来建立小标题。
        博客**必须**直接以#博客标题开始撰写。
        博客**必须**按图表引用规则引用论文最主要、最重要的至少一张图片（文章不包含图片或不包含主要框架图片除外），但也要注意详略得当，不要引用一些无关紧要的图片。
        博客**尽量**注意详略得当，文章花大篇幅介绍的地方你也要着重介绍。
        
        **必须遵守**图表引用规则：
        引用论文中的重要图表时，使用以下格式（注意千万不要出现不存在的Figure序号）：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X和Y替换为实际上的数字，其余的不可变动。

        **注意**：
        图片的命名是严格格式化的，一定是：
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX.png)<br>
        或者
        <br>![图X：简短说明]({self.data_path}/{arxiv_id}_FigureX(Y).png)<br>
        X代表数字或数字+括号数字，比如Figure2代表图2，Figure2(1)代表图2中的第一个子图，可能文中的描述是Figure2(a)，但不管怎样你在调用时需要将其转化为数字，比如第x个子图的索引就是Figure2(x)。
        注意F要大写。

        **必须不能**引用文中不存在的图片
        **必须不能**图片引用写成了Algorithm或http开头的网址等，比如{self.data_path}/{arxiv_id}_AlgorithmX.png。
        
        **示例**
        一些正确的图片引用示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(1).png)<br>
        错误图片命名示例：
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Algorithm2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_Figure2(a).png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_figure2.png)<br>
        <br>![图2：简短说明]({self.data_path}/{arxiv_id}_title2.png)<br>
        <br>![图2：简短说明](https://github.com/jeff-zhuo/paper_figure_captions/blob/main/{self.data_path}/{arxiv_id}_Figure5.png)<br>
        """

        print(F"正在为{arxiv_id}生成博客...")
         # 调用API生成内容并保存
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                prompt
            ]
        )
        
        # 保存为Markdown文件
        markdown_path = os.path.join(self.output_path, f"{arxiv_id}_问答.md")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(response.text)
    