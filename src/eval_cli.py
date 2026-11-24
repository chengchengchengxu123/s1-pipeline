import json
import argparse
import time
import os
import asyncio
from pathlib import Path

# ================= 1. 数据加载 =================
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

# ================= 2. Mock 模型 =================
async def async_mock_provider(question):
    await asyncio.sleep(0.05) # 模拟 50ms 网络延迟
    return f"Mock回答: {question[:10]}..."

# ================= 3. 单条任务执行 (重试+测速) =================
async def run_one_async(sem, sample, max_retries=3):
    sid = sample.get("qid", "unknown")
    q = sample.get("question", "")
    
    start_time = time.perf_counter()
    
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                # 💡 核心：在这里精准狙击 test-002 和 test-003 强制报错
                if sid in ["test-002", "test-003"]:
                    raise TimeoutError("模拟 API 限流超时")
                    
                ans = await async_mock_provider(q)
                status, error = "ok", None
                break # 成功则跳出重试循环
                
            except Exception as e:
                if attempt == max_retries:
                    ans, status, error = None, "error", f"重试{max_retries}次失败: {e}"
                else:
                    backoff_time = 2 ** attempt
                    print(f"⚠️ {sid} 触发重试，等待 {backoff_time}s 后第 {attempt+1} 次重试...")
                    await asyncio.sleep(backoff_time)
                
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    return {
        "sample_id": sid, "status": status, "output": ans, 
        "error": error, "latency_ms": elapsed_ms
    }

# ================= 4. 主控执行 (断点续跑+自动建目录) =================
async def run_eval_async(input_path, output_dir, concurrency):
    samples = load_samples(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 动态生成文件名
    run_id = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(input_path).stem 
    output_file = Path(output_dir) / f"{dataset_name}_result_{run_id}.jsonl"
    
    # 断点续跑逻辑
    done_ids = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("status") == "ok":
                    done_ids.add(row["sample_id"])

    pending_samples = [s for s in samples if s.get("qid") not in done_ids]
    
    if not pending_samples:
        print("🎉 所有任务均已完成，无需重跑！")
        return

    print(f"🚀 开始评测，待跑 {len(pending_samples)} 条，并发数={concurrency}...")
    
    sem = asyncio.Semaphore(concurrency)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        tasks = [run_one_async(sem, s) for s in pending_samples]
        for future in asyncio.as_completed(tasks):
            result = await future
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush() # 强制落盘

    print(f"✅ 评测完成！结果保存在 {output_file}")
    print(f"👉 下一步运行: uv run python -m src.eval_cli --mode report --report_file {output_file}")

# ================= 5. 数据质检 =================
def validate_data(input_path):
    print(f"🔍 开始校验数据: {input_path}")
    seen_ids = set()
    errors = []
    n_total = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            n_total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"行 {line_no}: JSON解析失败 - {e}")
                continue
            qid = data.get("qid")
            if not qid: errors.append(f"行 {line_no}: 缺失 qid")
            elif qid in seen_ids: errors.append(f"行 {line_no}: 重复 qid '{qid}'")
            else: seen_ids.add(qid)

    print("\n" + "="*40)
    print("🛡️ 数据质检报告")
    print("="*40)
    print(f"总行数: {n_total}, 合法唯一ID数: {len(seen_ids)}")
    if errors:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for err in errors[:10]: print(f"  - {err}")
    else:
        print("✅ 数据格式完美，无重复，无缺失！")
    print("="*40 + "\n")

# ================= 6. 汇总报告 (去重+P50/P95) =================
def summarize_report(result_file):
    if not os.path.exists(result_file):
        print(f"❌ 找不到结果文件: {result_file}")
        return
        
    latest_results = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            sid = row.get("sample_id")
            
            # 💡 核心逻辑：字典去重，后来的结果覆盖之前的
            if sid not in latest_results or row["status"] == "ok":
                latest_results[sid] = row

    n_total = len(latest_results)
    n_ok = sum(1 for r in latest_results.values() if r["status"] == "ok")
    n_error = n_total - n_ok
    error_rate = n_error / n_total if n_total > 0 else 0.0
    latencies = [r["latency_ms"] for r in latest_results.values() if r["status"] == "ok" and "latency_ms" in r]
    
    print("\n" + "="*40)
    print(f"📊 评测报告: {os.path.basename(result_file)}")
    print("="*40)
    print(f"总样本数 (n_total)  : {n_total}")
    print(f"成功数 (n_ok)       : {n_ok}")
    print(f"失败数 (n_error)    : {n_error}")
    print(f"失败率 (error_rate) : {error_rate:.2%}")
    
    if latencies:
        # 纯 Python 计算分位数，不需要 numpy
        latencies.sort()
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        print(f"P50 延迟 (中位数)   : {latencies[p50_idx]:.2f} ms")
        print(f"P95 延迟 (尾部)     : {latencies[p95_idx]:.2f} ms")
    else:
        print("P50/P95 延迟        : 无数据")
    print("="*40 + "\n")

# ================= 7. CLI 入口 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "report", "validate"], default="run")
    parser.add_argument("--input", default="data/test/questions.jsonl")
    parser.add_argument("--output_dir", default="artifacts/eval_runs")
    parser.add_argument("--report_file")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "run":
        asyncio.run(run_eval_async(args.input, args.output_dir, args.concurrency))
    elif args.mode == "report":
        if not args.report_file: print("❌ 报错：必须提供 --report_file 参数")
        else: summarize_report(args.report_file)
    elif args.mode == "validate":
        validate_data(args.input)