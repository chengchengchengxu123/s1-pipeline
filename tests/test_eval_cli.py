import pytest
import json
import os
from src.eval_cli import validate_data, summarize_report

# 测试 1：数据质检模块能否抓到重复 ID
def test_validate_data_catches_duplicates(tmp_path, capsys):
    test_file = tmp_path / "test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write('{"qid": "1", "question": "a"}\n')
        f.write('{"qid": "1", "question": "b"}\n') # 故意重复
        
    validate_data(str(test_file))
    captured = capsys.readouterr()
    assert "发现 1 个错误" in captured.out
    assert "重复 qid '1'" in captured.out

# 测试 2：报告汇总能否正确计算失败率与 P95 延迟
def test_summarize_report_metrics(tmp_path, capsys):
    result_file = tmp_path / "result.jsonl"
    # 构造 10 条数据：9条成功（延迟100ms），1条失败
    with open(result_file, "w", encoding="utf-8") as f:
        for i in range(9):
            f.write(json.dumps({"sample_id": f"s{i}", "status": "ok", "latency_ms": 100.0}) + "\n")
        f.write(json.dumps({"sample_id": "s9", "status": "error", "error": "timeout"}) + "\n")

    summarize_report(str(result_file))
    captured = capsys.readouterr()
    assert "总样本数 (n_total)  : 10" in captured.out
    assert "失败率 (error_rate) : 10.00%" in captured.out
    assert "P50 延迟 (中位数)   : 100.00 ms" in captured.out
    assert "P95 延迟 (尾部)     : 100.00 ms" in captured.out