import os
import json
import time
import random  # 补上这个
from openai import OpenAI

class QwenProvider:
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未找到环境变量 DASHSCOPE_API_KEY，请检查 ~/.bashrc 配置。")
        
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model="qwen3.8-flash"

    def chat(self, prompt, expect_json=False, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0
                )
                content = response.choices[0].message.content
                
                if expect_json:
                    try:
                        clean_content = content.strip(" \n`")
                        if clean_content.startswith("json"):
                            clean_content = clean_content[4:]
                        return json.loads(clean_content)
                    except json.JSONDecodeError:
                        raise ValueError(f"模型返回的并非标准 JSON: {content}")
                
                return content
                
            except Exception as e:
                print(f"[警告] 第 {attempt + 1} 次请求失败: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"超过最大重试次数，任务失败。最后错误: {e}")
                
                wait_time = 5.0 
                print(f"   -> 等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)

# === 新增的 Mock 假模型 ===
class MockProvider:
    def __init__(self):
        self.step = 0  # 增加一个计数器

    def chat(self, prompt, expect_json=False, max_retries=3):
        print("[Mock模式] 正在返回模拟数据...")
        if expect_json:
            self.step += 1
            if self.step == 1:
                # 第一次：模拟模型决定调用工具
                return {"action": "query_metadata", "args": {"name": "questions"}}
            else:
                # 第二次：模型看到了 Observation，决定输出最终回答
                return {"action": "final_answer", "answer": "经过查询，questions.json 共有 10 条记录。"}
        return "这是一个模拟的大模型回答..."

# === 新增的工厂函数 ===
def get_provider(use_mock=True):
    if use_mock:
        return MockProvider()
    return QwenProvider()