from openai import OpenAI
import os


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if not api_key:
        api_key = your api key
    return OpenAI(api_key=api_key)


def chat(prompt, pdf_url, model: str = "gpt-4o-mini"):
    client = _get_client()
    response = client.responses.create(
        model="gpt-4o-mini",
        temperature=0.1,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_file",
                        "file_url": pdf_url,
                    },
                ],
            }
        ],
    )
    return response.output_text


def chat_simple(prompt: str, model: str = "gpt-4o") -> str:
    client = _get_client()
    response = client.responses.create(
        model=model,
        temperature=0.1,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return response.output_text


#TODO：
# 
# Gemini
# deepseek-3.1
# openai oss





if __name__ == "__main__":
    a = chat("总结这篇论文。", "https://openreview.net/pdf?id=wCUw8t63vH")
    print(a)