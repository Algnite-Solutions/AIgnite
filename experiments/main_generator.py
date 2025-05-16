from google import genai
from google.genai import types
import os

client = genai.Client()

# Path to the local PDF file

#html_path = "../test/htmls/2501.11216.html"
# 3354 tokens as pdf
# 91165 tokens as html

def generate_single_blog(pdf_path):
    # Read and encode the PDF bytes
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = pdf_file.read()
    arxiv_id = ".".join(pdf_path.split("/")[-1].split(".")[:-1])
    data_path = "../test/tem"
    print(arxiv_id)
    #prompt = "生成一篇中文解读博客, 适当的引用图表, 目标用户是对这个领域有了解的研究人员"
    prompt = f"""
    You're generating a mark down blog post summarizing a paper with arXiv ID {arxiv_id} for researchers in the field. The style is similar to medium science blog.

    In your blog, you can cite a **few of the most important figures** from the paper (ideally no more than 3) to help understanding. For each selected figure, render it as a standalone Markdown image:
      `<br>![Figure X: short caption]({data_path}/{arxiv_id}_FigureX.png)<br>`

    Do **not** use inline figure references like “as shown in Figure 2”. Do **not** cite tables.
    Start directly with Blog title.
    """

    response = client.models.generate_content(
      model="gemini-2.5-flash-preview-04-17",
      contents=[
          types.Part.from_bytes(
            data=pdf_data,
            mime_type='application/pdf',
          ),
          prompt])

    markdown_path = f"./experiments/output/{arxiv_id}.md"
    with open(markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(response.text)

    print(f"Markdown file saved to {markdown_path}")
    print(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)


pdf_folder = "./test/pdfs/"
for pdf_file in os.listdir(pdf_folder):
    generate_single_blog(os.path.join(pdf_folder, pdf_file))