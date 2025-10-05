from deepseek import chat_deepseek
from ChatGPT import chat, chat_simple
import json
import re

def process_single(blog_path, paper_path):

    blog_text = open(blog_path, "r").read()
    paper_text = open(paper_path,"r").read()
    pdf_url = "https://arxiv.org/pdf/2404.07738v2.pdf"

    # 中文提示：中性比对要求与中文 JSON 输出（强化细节一致性核查与证据定位）
    prompt = f"""
你现在需要进入深度思考模式，执行一项严谨的论文博客幻觉漏洞检测任务。请严格遵循以下步骤与要求，逐环节开展分析：

1. 明确检测对象与依据：以提供的「原论文全文（文本格式）」「原论文URL」作为唯一事实依据，对比由其他大模型生成的「论文解读博客」，找出博客中与原论文不符的幻觉内容。

2. 深度思考执行规则：

◦ 采用「逐段精读」方式处理博客内容，每读完一段后立即暂停，启动事实核查流程。

◦ 对博客中以下关键信息点进行强制校验，必须回归原论文对应位置进行比对，不允许凭记忆或推断判断：

◦ 模型名称、方法名称（如算法、框架名称）

◦ 所有数字数据（性能指标百分比、数据集规模、实验参数、延迟数值、对话轮次等）

◦ 学术机构名称、作者姓名及合作关系

◦ 核心实验结论、结论对比依据

3. 漏洞分级与定义：

重要漏洞：直接影响学术事实准确性、误导读者对论文核心价值判断的内容，包括但不限于：

◦ 虚构作者合作关系、机构隶属关系

◦ 捏造或篡改性能指标（如准确率、召回率、效率提升百分比等）

◦ 虚构实验数据（如数据集样本量、训练迭代次数、对比模型数量等）

◦ 错误的延迟数据及延迟对比结果

◦ 错误的对话轮次、响应时间等关键实验结果

轻微漏洞：不影响论文核心结论，但存在细节偏差的内容，包括但不限于：

◦ 地名、机构简称等非核心信息错误

◦ 虚构或错误的论文引用链接、附录链接

◦ 学术术语用词不当（如将「预训练模型」误写为「微调模型」）

◦ 结论限定条件缺失（如省略原论文中「在特定数据集上」的前提）

4. 输出格式要求：
输出内容需完全符合 JSON 格式规范，确保无语法错误（如引号闭合、逗号正确、键值对匹配），不允许添加任何 JSON 之外的注释或说明文本。
JSON 结构需包含 3 个一级键：detection_overview（检测概述）、critical_vulnerabilities（重要漏洞集合）、minor_vulnerabilities（轻微漏洞集合），具体字段定义与示例如下：

```
{{
  "detection_overview": {{
    "total_blog_paragraphs": "检测的博客总段落数（填写具体数字，如12）",
    "critical_vulnerability_count": "发现的重要漏洞总数（填写具体数字，如3）",
    "minor_vulnerability_count": "发现的轻微漏洞总数（填写具体数字，如5）",
    "verification_basis": "说明核查依据，如「基于原论文全文（文本格式）、原论文PDF及原论文URL（XXX）进行逐点比对」"
  }},
  "critical_vulnerabilities": [
    {{
      "vulnerability_type": "漏洞类型（从重要漏洞定义中选择，如「捏造性能指标」）",
      "blog_content": "博客中存在漏洞的原文内容（需完整引用对应句子，加引号，如「该模型在测试集上的准确率高达98%」）",
      "original_paper_content": "原论文中对应的事实内容（完整引用原论文相关句子/数据，加引号，如「该模型在测试集上的准确率为89.2%」）",
      "deviation_description": "清晰说明偏差本质与影响，如「虚构8.8个百分点的准确率提升，严重夸大模型性能，误导对模型效果的判断」"
    }},
    {{
      "vulnerability_type": "错误合作关系",
      "blog_content": "「该研究由XX大学与YY实验室联合完成」",
      "original_paper_content": "「该研究团队所有成员均隶属于XX大学」",
      "deviation_description": "捏造YY实验室的合作参与情况，错误表述研究机构背景"
    }}
  ],
  "minor_vulnerabilities": [
    {{
      "vulnerability_type": "用词不当",
      "blog_content": "「模型通过监督学习完成预训练阶段」",
      "original_paper_content": "「模型通过自监督学习策略完成预训练」",
      "deviation_description": "混淆核心训练方法术语，将「自监督学习」误写为「监督学习」，但不影响对模型整体流程的理解"
    }},
    {{
      "vulnerability_type": "链接虚构",
      "blog_content": "「论文附录细节可访问https://fakeurl.com/appendix查看」",
      "original_paper_content": "「本论文未提供公开附录链接」",
      "deviation_description": "虚构不存在的附录访问链接，属于非核心信息偏差"
    }}
  ]
}}
```

若某类漏洞数量为 0（如无重要漏洞），需将对应数组设为空数组（如"critical_vulnerabilities": []），不可删除该键或留空值；所有字段需填写具体内容，不允许保留占位符文本（如删除示例中的括号与说明文字，替换为实际检测结果）。

请基于上述要求，严谨执行检测任务，确保所有关键信息点均经过原论文交叉验证，漏洞分级与描述准确无误。

需要我帮你把这个Prompt保存为文档格式，或者根据你提供的具体论文和博客内容，先模拟一次检测流程作为示例吗？
博客：
{blog_text}
论文原文：
{paper_text}
"""

    response = chat_deepseek(prompt)
    print(response)
    return response

if __name__ == "__main__":
    process_single("/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/single_blog_and_pdf/2404.07738v2.md",
    "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/single_blog_and_pdf/2404.07738v2.json")