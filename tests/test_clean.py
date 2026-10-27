import pytest
import json
from src.clean import validate_record, MissingFieldError, load_lines

# 测试 1：正常数据（什么都不该抛）
def test_valid_record():
    # 加上 timestamp，让它符合新的校验规则
    record = {"doc_id": "1", "source": "web", "content": "hello", "timestamp": "2023-10-27"}
    validate_record(record)

# 测试 2：缺字段（必须抛 MissingFieldError）
def test_missing_field():
    record = {"doc_id": "1", "source": "web"}
    with pytest.raises(MissingFieldError):
        validate_record(record)

# 测试 3：字段为空（必须抛 MissingFieldError）
def test_empty_content():
    record = {"doc_id": "1", "source": "web", "content": "   "}
    with pytest.raises(MissingFieldError):
        validate_record(record)

# 测试 4：坏 JSON 格式（测解析过程）
def test_bad_json():
    line = '{"doc_id": "1", "source": "web"'
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)

# 测试 5：空文件读取（生成器不应报错，应返回空）
def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    lines = list(load_lines(empty_file))
    assert len(lines) == 0