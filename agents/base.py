import os
from openai import OpenAI

# 统一配置 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=60.0,
    max_retries=3
)

def ask_llm(system_prompt: str, user_content: str) -> str:
    """统一的 LLM 调用接口"""
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content