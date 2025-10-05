from ChatGPT import chat, chat_simple
import json
import re

def main():

    blog_path = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/single_blog_and_pdf/2404.07738v2.md"
    blog_text = open(blog_path, "r").read()
    paper_text = open("/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/single_blog_and_pdf/2404.07738v2.json","r").read()
    pdf_url = "https://arxiv.org/pdf/2404.07738v2.pdf"

    # 中文提示：中性比对要求与中文 JSON 输出（强化细节一致性核查与证据定位）
    prompt = f"""
你是一个论文博客幻觉检测助手。我会提供一篇中文博客内容和该论文的PDF链接。
请你对比pdf和博客进行分析，然后以json的形式指出有什么严重错误、中度问题和轻微问题。
这三个程度的幻觉定义以及示例如下：

检测到的幻觉和错误
严重错误 🔴
1. 虚构的合作关系
例如：博客声明： "京东和沃尔玛的研究团队提出的AdaptJobRec系统" 论文事实： 作者来自Walmart Global Tech和University of Arkansas，没有京东(JD)参与 严重程度： Level 4 - 完全虚构的信息
2. 捏造的数据
例如：博客声明： "这种精确的记忆提取使规划器的子任务分解准确率提升17%" 论文事实： 论文中没有提到17%这个数字 严重程度： Level 4 - 凭空捏造的数据
3. 虚构的性能指标
例如：博客声明： "并行化策略使复杂查询的处理效率提升38%" 论文事实： 论文未提及38%的效率提升 严重程度： Level 4 - 完全虚构
4. 捏造的优化数据
例如：博客声明： "Redis缓存机制减少62%的LLM调用" 论文事实： 论文提到Redis缓存但没有62%这个具体数字 严重程度： Level 4 - 虚构数据
5. 虚构的比较数据
例如：博客声明： "较LLM基线模型提升35%" 论文事实： 论文未提及35%的提升幅度 严重程度： Level 4 - 捏造的比较数据
6. 错误的延迟比较
例如：博客声明： "响应延迟仅比最快速的Frequency方法高8%" 论文事实： Frequency方法延迟0.32秒，AdaptJobRec为0.36秒，差异为12.5%，不是8% 严重程度： Level 3 - 计算错误
7. 错误的对话轮次数据
例如：博客声明： "RAG基线的6.7次" 论文事实： RAG基线是7.10次，不是6.7次 严重程度： Level 3 - 数值错误
8. 错误的响应延迟数据
例如：博客声明： "响应延迟从1.09秒压缩到0.498秒" 论文事实： RAG延迟是1065毫秒(1.065秒)，不是1.09秒 严重程度： Level 2 - 数值不准确
中度问题 🟡
9. 地名错误
例如：博客声明： "比较西雅图和阳光谷的岗位数量" 论文事实： 论文中是"Sunnyvale"（森尼维尔），不是"阳光谷" 严重程度： Level 2 - 翻译错误
10. 虚构的链接
例如：博客声明： 提供了GitHub和文档链接 论文事实： 论文中没有提供这些链接 严重程度： Level 3 - 虚构的资源链接
轻微问题 🟢
11. 标题限定词缺失
例如：博客声明： "首个集成个性化推荐算法的对话式求职系统" 论文事实： "the first conversational job recommendation system" 问题： 虽然保留了"对话式"，但"首个"的限定范围可能被误解 严重程度： Level 1

那么，请你参考上面的标准，没有则不用提及，有的话则说明，并最后计算出各个等级幻觉的数量。
请你仔细检查blog的每个句子。
注意不要错把我给你的例子当成真实的漏洞，你的漏洞检测应该完全来自于下面我提供给你的博客和论文原文。
博客：
{blog_text}
论文原文：
{paper_text}
"""

    response = chat_simple(prompt)
    print(response)

if __name__ == "__main__":
    main()