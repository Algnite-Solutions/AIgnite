from ChatGPT import chat, chat_simple
import json
import re

def main():

    blog_path = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/single_blog_and_pdf/2404.07738v2.md"
    blog_text = open(blog_path, "r").read()
    pdf_url = "https://arxiv.org/pdf/2404.07738v2.pdf"

    # 中文提示：中性比对要求与中文 JSON 输出（强化细节一致性核查与证据定位）
    prompt = f"""
你是一个论文博客幻觉检测助手。我会提供一篇中文博客内容和该论文的PDF链接。

任务要求：
1) 逐段推理分析博客内容，与论文原文逐项比对；
2) 从博客中“抽取所有事实性断言”，并逐项核验是否被论文明确支持；识别所有不一致并归因，覆盖以下维度（保持中性，不要假设具体错误线索）：
   - 方法/流程：是否与论文方法步骤一致，是否多写或漏写关键环节；
   - 模型：名称/版本/规模、是否为论文实际使用或评估对象；
   - 数据：数据集名称/来源、样本规模、划分方式（train/val/test）、预处理与过滤；
   - 实验设置：超参数（学习率、批大小、训练轮数等）、随机种子、硬件/加速器、实现细节；
   - 评测：指标名称（如Accuracy、F1、BLEU等）的定义、报告的具体数值及其对应图表/表格；
   - 结果与结论：是否准确反映论文的发现、限制与适用范围；
   - 图表与编号：引用的图（Figure）/表（Table）编号及文字是否与论文一致；
   - 数值与量纲：任何具体数字、百分比、误差范围、置信区间等是否与论文一致。
   覆盖提示：
   - 抽取并列出博客中出现的“全部模型名称/版本”清单（如包含 GPT/LLM/模型家族名称及后缀：mini/pro/turbo/数字版本等），逐一在论文中核验是否真实出现且语境一致；若未出现或语义不符，则单独输出为一条记录；
   - 特别留意具体数据集名、带单位的数值（%, ×, ms, MB 等）、图表编号（Figure/Table + 数字）并核验其在论文中的对应；
3) 输出为先按照上述模式进行的思考（包括对每一段的分析和思考，以及疑问和验证），然后输出JSON数组，字段与示例如下，且必须用中文写作：

（前面需要是对博客的每一段的分析，疑问，验证和思考。）
[
  {{
    "id": 1,
    "type": "严重幻觉" 或 "轻微幻觉",
    "blog_content": "指出博客中的原句或关键片段（必须中文）",
    "original_paper_meaning": "对应论文原意/事实（英文）",
    "reasoning": "你的比对与推理说明（中文，可引用具体页码/节标题/图表编号作为证据）",
    "paper_evidence": "从论文中摘录的关键原文（英文，尽量精确引句）",
    "paper_location": "证据大致位置（如第X页/第Y节/Figure Z/Table T）"
  }}
]

判定标准：
- 严重幻觉：与论文事实明显矛盾，足以误导读者（如模型/数据/方法/结论描述错误）。
- 轻微幻觉：表述不严谨、细节偏差但不改变核心结论。
博客：
{blog_text}
论文PDF：
{pdf_url}
"""

    response = chat(prompt, pdf_url)
    print(response)

if __name__ == "__main__":
    main()