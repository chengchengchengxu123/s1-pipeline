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
    time.sleep(0.01) # 模拟网络延迟
    return f"这是 Mock 回答：{question[:10]}..."

# 3. 执行评测的核心逻辑
def run_eval(input_path, output_dir):
    samples = load_samples(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"mock_run_{run_id}.jsonl"
    config_file = Path(output_dir) / f"mock_run_{run_id}_config.json"
    
    config = {"run_id": run_id, "n_samples": len(samples), "provider": "mock"}
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"🚀 开始评测，共 {len(samples)} 条数据...")
    results = []
    
    for sample in samples:
        sid = sample.get("qid", "unknown")
        q = sample.get("question", "")
        try:
            # 💡 为了测试报告功能，我们强制让 sample_id 为 "2" 的数据报错
            if sid == "test-002":
                raise TimeoutError("人为制造的 API 超时")
            
            ans = mock_provider(q)
            results.append({"sample_id": sid, "status": "ok", "output": ans})
        except Exception as e:
            # 失败被隔离，不会导致整体崩溃
            results.append({"sample_id": sid, "status": "error", "error": str(e)})
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"✅ 评测完成！结果已保存至 {output_file}")
    print(f"👉 下一步，你可以运行: uv run python -m src.eval_cli --mode report --report_file {output_file}")

# 4. 统计报告核心逻辑（今天的重点）
def summarize_report(result_file):
    if not os.path.exists(result_file):
        print(f"❌ 找不到结果文件: {result_file}")
        return
        
    n_total = 0
    n_ok = 0
    n_error = 0
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            n_total += 1
            if row.get("status") == "ok":
                n_ok += 1
            else:
                n_error += 1
                
    error_rate = n_error / n_total if n_total > 0 else 0.0
    
    print("\n" + "="*40)
    print(f"📊 评测报告: {os.path.basename(result_file)}")
    print("="*40)
    print(f"总样本数 (n_total)  : {n_total}")
    print(f"成功数 (n_ok)       : {n_ok}")
    print(f"失败数 (n_error)    : {n_error}")
    print(f"失败率 (error_rate) : {error_rate:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI 评测工具")
    parser.add_argument("--mode", choices=["run", "report"], default="run", help="run=跑评测，report=出报告")
    parser.add_argument("--input", default="data/test/questions.jsonl", help="输入 JSONL 路径")
    parser.add_argument("--output_dir", default="artifacts/eval_runs", help="输出目录")
    parser.add_argument("--report_file", help="当 mode=report 时，指定要分析的 jsonl 文件路径")
    
    args = parser.parse_args()
    
    if args.mode == "run":
        run_eval(args.input, args.output_dir)
    elif args.mode == "report":
        if not args.report_file:
            print("❌ 报错：使用 report 模式必须提供 --report_file 参数")
        else:
            summarize_report(args.report_file)