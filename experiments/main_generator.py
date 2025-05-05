from google import genai
from google.genai import types

client = genai.Client()

# Path to the local PDF file
pdf_path = "./data/2501.11216.pdf"
html_path = "../test/htmls/2501.11216.html"
# 3354 tokens as pdf
# 91165 tokens as html

with open(html_path, "rb") as html_file:
    html_data = html_file.read()

# Read and encode the PDF bytes
with open(pdf_path, "rb") as pdf_file:
    pdf_data = pdf_file.read()
arxiv_id = ".".join(pdf_path.split("/")[-1].split(".")[:-1])
data_path = "/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem"
print(arxiv_id)
#prompt = "生成一篇中文解读博客, 适当的引用图表, 目标用户是对这个领域有了解的研究人员"
prompt = f"""
You're generating a blog post summarizing a paper with arXiv ID {arxiv_id}.

Only cite **important figures** that help readers understand the main contributions or results. Do **not** cite all figures — include only those that are essential to the blog post.

When citing a figure:
- Use the exact figure number from the paper.
- Use the following Markdown format:
  
  `[Figure X]({data_path}/{arxiv_id}_FigureX): <short, descriptive caption>`

Avoid vague references like “the figure below” or “as shown above”.

Do **not** cite any tables. Your output should include only relevant **figure references** and descriptions.
"""

response = client.models.generate_content(
  model="gemini-2.5-flash-preview-04-17",
  contents=[
      types.Part.from_bytes(
        data=pdf_data,
        mime_type='application/pdf',
      ),
      prompt])

markdown_path = "./generated_blog.md"
with open(markdown_path, "w", encoding="utf-8") as md_file:
    md_file.write(response.text)

print(f"Markdown file saved to {markdown_path}")

