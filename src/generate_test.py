import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载 Tokenizer 和 模型
model_name = "distilgpt2"
print(f"正在加载模型 {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 2. 测试 Tokenizer (编码与解码)
print("\n--- 测试 Tokenizer ---")
text = "I love AI."  # 注意：distilgpt2 是英文模型，用中文效果不好，这里先用英文
ids = tokenizer.encode(text)
print(f"原文本: '{text}'")
print(f"Token IDs: {ids}")
print(f"解码回文本: '{tokenizer.decode(ids)}'")
print(f"词表大小 (vocab_size): {tokenizer.vocab_size}")
print(f"上下文窗口 (max_position_embeddings): {model.config.max_position_embeddings}")

# 3. 测试生成参数对比
print("\n--- 测试生成参数 (对比随机性) ---")
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 定义不同的生成配置
configs = [
    {"name": "贪心解码 (确定性)", "do_sample": False},
    {"name": "低温采样 (保守)", "do_sample": True, "temperature": 0.3},
    {"name": "高温采样 (胡说八道)", "do_sample": True, "temperature": 1.5},
    {"name": "Top-p 0.85 (平衡)", "do_sample": True, "temperature": 1.0, "top_p": 0.85},
]

for cfg in configs:
    name = cfg.pop("name") # 取出名字，剩下的参数用于生成
    print(f"\n[{name}]")
    
    # 每个配置跑两次，验证输出是否一致
    for i in range(2):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=15, 
                pad_token_id=tokenizer.eos_token_id,
                **cfg
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  生成{i+1}: {generated_text}")