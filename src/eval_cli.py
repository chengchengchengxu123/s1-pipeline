import json
import argparse
import time
import os
from pathlib import Path

# 1. 读取数据，捕获坏 JSON
def load_samples(filepath):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"数据解析失败 {filepath}:{line_no} - {e}")
    return samples

# 2. Mock Provider（模拟模型调用）
def mock_provider(question):
    time.sleep(0.1) # 模拟 100ms 网络延迟
    if "异常" in question: # 故意留个口子测试异常
        raise TimeoutError("模拟 API 超时")
    return f"这是 Mock 回答：{question[:10]}..."

# 3. 执行评测的核心逻辑
def run_eval(input_path, output_dir):
    samples = load_samples(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"mock_run_{run_id}.jsonl"
    config_file = Path(output_dir) / f"mock_run_{run_id}_config.json"
    
    # 保存配置，保证后续可追溯
    config = {
        "run_id": run_id,
        "input_path": str(input_path),
        "n_samples": len(samples),
        "provider": "mock"
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"🚀 开始评测，共 {len(samples)} 条数据...")
    results = []
    
    for sample in samples:
        sid = sample.get("sample_id", "unknown")
        q = sample.get("question", "")
        
        try:
            # 真实情况下这里会调用你的真实 API 或 Agent
            ans = mock_provider(q)
            results.append({"sample_id": sid, "status": "ok", "output": ans})
        except Exception as e:
            # 核心原理：失败绝不能被当作 0 分或静默丢弃
            results.append({"sample_id": sid, "status": "error", "error": str(e)})
            
    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"✅ 评测完成！结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI 评测工具")
    parser.add_argument("--input", default="data/questions.jsonl", help="输入 JSONL 路径")
    parser.add_argument("--output_dir", default="artifacts/eval_runs", help="输出目录")
    
    args = parser.parse_args()
    run_eval(args.input, args.output_dir)