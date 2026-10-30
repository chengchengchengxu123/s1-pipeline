from src.provider import get_provider

def main():
    # 这里改成 True 用 Mock 测试，改回 False 用真实 API
    provider = get_provider(use_mock=True) 
    
    print("初始化 Provider 完毕...")
    
    # 测试1：普通文本请求
    print("\n--- 测试普通文本 ---")
    ans = provider.chat("用一句话解释什么是大语言模型。")
    print(f"普通回答: {ans}\n")
    
    # 测试2：结构化 JSON 请求
    print("\n--- 测试 JSON 输出 ---")
    prompt = '请输出一个 JSON，包含字段: {"name": "张三", "age": 25}。只输出 JSON，不要其他文字。'
    json_data = provider.chat(prompt, expect_json=True)
    print(f"解析后的字典: {json_data}")
    print(f"获取 name 字段: {json_data['name']}")

if __name__ == "__main__":
    main()