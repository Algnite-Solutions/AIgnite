# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-7d1b4bfa589c45f9a352d3e22623eec1",
    base_url="https://api.deepseek.com")

def chat_deepseek(prompt):
    response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
            {"role": "system", "content": "You need to engage in deep thinking."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    #print(f"Model is: {response.model}")
    #print(f"Output is: {response.choices[0].message.content}")
    print(response.choices[0].message.reasoning_content)
    return response.choices[0].message.content

#chat_deepseek("hi,请问爱因斯坦老师的第一个学生是谁")