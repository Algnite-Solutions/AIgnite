import json
import os
from unittest import expectedFailure
from single_test_v3 import process_single

def process_batch(json_folder, blog_folder, output_file):
    for root, dirs, json_files in os.walk(json_folder):
        for file in json_files:
            if file.endswith('.json'):
                filename_no_ext = os.path.splitext(file)[0]
                json_path = os.path.join(root, file)
                md_path = os.path.join(blog_folder, f"{filename_no_ext}.md")

                try:
                    response = process_single(md_path, json_path)
                except Exception as e:
                    print("错误。")
                    response = ""

                # 构造要保存的对象
                item = {
                    "filename_no_ext": filename_no_ext,
                    "response": response  # 纯文本
                }

                # 追加写入一行 JSON
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    json.dump(item, f_out, ensure_ascii=False)
                    f_out.write('\n')

                print("-" * 20)

if __name__ == "__main__":
    '''json_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/jsons"
    blog_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/blog_TLDR_Gemini"
    output_file = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/result_Gemini.jsonl"  # 后缀建议用 .jsonl
    process_batch(json_folder, blog_folder, output_file)'''
    json_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/jsons"
    blog_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/blog_TLDR_Qwen"
    output_file = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/result_Qwen.jsonl"  # 后缀建议用 .jsonl
    process_batch(json_folder, blog_folder, output_file)

    json_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/jsons"
    blog_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/blog_TLDR_Gemini_pdf"
    output_file = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/result_Gemini_pdf.jsonl"  # 后缀建议用 .jsonl
    process_batch(json_folder, blog_folder, output_file)

    json_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/jsons"
    blog_folder = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/BLOG/blog_TLDR_QwenBIG"
    output_file = "/data3/guofang/peirongcan/AIgnite/experiments/hallucination_test/data/result_QwenBIG.jsonl"  # 后缀建议用 .jsonl
    process_batch(json_folder, blog_folder, output_file)
    