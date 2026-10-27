import json
from pathlib import Path

# 指定要读取的文件路径
path = Path("data/sample.jsonl")

print("开始检查数据...")
# 逐行读取并处理
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        # 尝试解析 JSON
        record = json.loads(line)
        # 检查必填字段是否缺失
        missing = [k for k in ("doc_id", "source", "content") if k not in record]
        if missing:
            print(f"第 {i} 行缺失字段: {missing}")
        else:
            print(f"第 {i} 行正常")
    except json.JSONDecodeError:
        # 如果 JSON 格式坏了，会被这里捕获
        print(f"第 {i} 行是坏 JSON 格式！")