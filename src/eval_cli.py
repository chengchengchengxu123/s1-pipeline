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
        
    latest_results = {} # 💡 用字典以 sample_id 为键进行去重
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            sid = row.get("sample_id")
            
            # 💡 核心逻辑：如果文件里有重复的 ID，用最新的覆盖旧的
            # 或者如果之前已经有 "ok" 了，就不要被后来的 "error" 覆盖（视业务而定）
            if sid not in latest_results or row["status"] == "ok":
                latest_results[sid] = row

    # 基于去重后的字典来统计
    n_total = len(latest_results)
    n_ok = sum(1 for r in latest_results.values() if r["status"] == "ok")
    n_error = n_total - n_ok
    error_rate = n_error / n_total if n_total > 0 else 0.0

    latencies = [r["latency_ms"] for r in latest_results.values() if r["status"] == "ok" and "latency_ms" in r]
    
    # ... (保留你下面的 print 打印逻辑不变) ...
    print("\n" + "="*40)
    print(f"📊 评测报告: {os.path.basename(result_file)}")
    print("="*40)
    print(f"总样本数 (n_total)  : {n_total}")
    print(f"成功数 (n_ok)       : {n_ok}")
    print(f"失败数 (n_error)    : {n_error}")
    print(f"失败率 (error_rate) : {error_rate:.2%}")
    
    if latencies:
        import numpy as np
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        print(f"P50 延迟 (中位数)   : {p50:.2f} ms")
        print(f"P95 延迟 (尾部)     : {p95:.2f} ms")
    else:
        print("P50/P95 延迟        : 无数据")
    print("="*40 + "\n")


import asyncio
import time
import json

# 1. 异步的 Mock Provider
async def async_mock_provider(question):
    await asyncio.sleep(0.5)  # 模拟 500ms 的真实网络请求
    if "test-002" in question:
        raise TimeoutError("模拟 API 超时")
    return f"Mock回答: {question[:5]}"

# 2. 单条数据的异步执行逻辑
async def run_one_async(sem, sample, max_retries=3):
    sid = sample.get("qid", "unknown")
    q = sample.get("question", "")
    
    start_time = time.perf_counter()
    
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                # 1. 模拟必然超时的测试数据
                if sid == "test-002":
                    raise TimeoutError("模拟 429 限流或网络超时")
                
                # 2. 正常请求
                ans = await async_mock_provider(q)
                status, error = "ok", None
                break # 成功就跳出重试循环
                
            except TimeoutError as e:
                if attempt == max_retries:
                    # 重试次数用尽，记录最终失败
                    ans, status, error = None, "error", f"重试{max_retries}次后失败: {e}"
                else:
                    # 💡 核心：指数退避 (1秒, 2秒, 4秒...)
                    backoff_time = 2 ** attempt 
                    print(f"⚠️ {sid} 触发重试，等待 {backoff_time}s 后第 {attempt+1} 次重试...")
                    await asyncio.sleep(backoff_time)
            except Exception as e:
                ans, status, error = None, "error", str(e)
                break
                
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {"sample_id": sid, "status": status, "output": ans, "error": error, "latency_ms": elapsed_ms}

# 3. 异步主控循环
async def run_eval_async(input_path, output_dir, concurrency):
    samples = load_samples(input_path)
    if len(samples) < 30:
        samples = (samples * 4)[:30]
        
    # 固定输出文件名，不要带时间戳，这样下次运行才能追加到同一个文件！
    output_file = Path(output_dir) / f"resume_test_run.jsonl"
    
    # 💡 核心 1：读取已完成的任务，存入 done_ids
    done_ids = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("status") == "ok":
                    done_ids.add(row["sample_id"])
        print(f"♻️ 发现历史记录，已跳过 {len(done_ids)} 条已成功的数据。")

    # 💡 核心 2：过滤出还需要跑的任务
    pending_samples = [s for s in samples if s.get("qid") not in done_ids]
    
    if not pending_samples:
        print("🎉 所有任务均已完成，无需重跑！")
        return

    print(f"🚀 开始续跑，待跑 {len(pending_samples)} 条，并发数={concurrency}...")
    
    sem = asyncio.Semaphore(concurrency)
    
    # 💡 核心 3：边跑边写文件，使用 'a' 追加模式，防止中途 Ctrl+C 丢失进度
    with open(output_file, 'a', encoding='utf-8') as f:
        tasks = [run_one_async(sem, s) for s in pending_samples]
        
        # as_completed 能在任何一个任务完成时立刻返回，而不是等全部完成
        for future in asyncio.as_completed(tasks):
            result = await future
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush() # 强制写入磁盘

    print(f"✅ 续跑完成！结果保存在 {output_file}")

    def summarize_report(result_file):
        # ... (保留你昨天的 n_total, n_ok 统计逻辑) ...
        
        latencies = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row["status"] == "ok":
                    latencies.append(row["latency_ms"])
                    
        if latencies:
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            print(f"P50 延迟: {p50:.2f} ms")
            print(f"P95 延迟: {p95:.2f} ms")

def validate_data(input_path):
    print(f"🔍 开始校验数据: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在")
        return

    seen_ids = set()
    errors = []
    n_total = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            n_total += 1
            
            # 1. 校验 JSON 格式
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"行 {line_no}: JSON解析失败 - {e}")
                continue
                
            # 2. 校验必要字段
            qid = data.get("qid")
            question = data.get("question")
            if not qid:
                errors.append(f"行 {line_no}: 缺失 qid 字段")
            if not question:
                errors.append(f"行 {line_no}: 缺失 question 字段")
                
            # 3. 校验 ID 唯一性
            if qid in seen_ids:
                errors.append(f"行 {line_no}: 发现重复的 qid '{qid}'")
            else:
                seen_ids.add(qid)

    # 打印质检报告
    print("\n" + "="*40)
    print("🛡️ 数据质检报告")
    print("="*40)
    print(f"总行数: {n_total}, 合法唯一ID数: {len(seen_ids)}")
    if errors:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for err in errors[:10]: # 只展示前10个
            print(f"  - {err}")
    else:
        print("✅ 数据格式完美，无重复，无缺失！")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI 评测工具")
    parser.add_argument("--mode", choices=["run", "report", "validate"], default="run", help="run=跑评测，report=出报告")
    parser.add_argument("--input", default="data/test/questions.jsonl", help="输入 JSONL 路径")
    parser.add_argument("--output_dir", default="artifacts/eval_runs", help="输出目录")
    parser.add_argument("--report_file", help="当 mode=report 时，指定要分析的 jsonl 文件路径")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数")
    args = parser.parse_args()
    
    if args.mode == "run":
        asyncio.run(run_eval_async(args.input, args.output_dir, args.concurrency))
    elif args.mode == "report":
        if not args.report_file:
            print("❌ 报错：使用 report 模式必须提供 --report_file 参数")
        else:
            summarize_report(args.report_file)
    elif args.mode == "validate":  # 💡 新增这个分支
        validate_data(args.input)
    
        
        