import json
import hashlib
from pathlib import Path

# 自定义异常：这是给 main() 看的信号
class MissingFieldError(Exception):
    pass

class BadJSONError(Exception):
    pass
def load_lines(path: Path):
    # 使用 yield 的生成器，文件多大都不会一次性读进内存
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过完全空白的行
                yield line.strip()
def validate_record(record: dict):

    for key in ("doc_id", "source", "content", "timestamp"):

        # 检查 key 是否存在，或者 value 是不是空字符串/纯空格
        if key not in record or not str(record[key]).strip():
            raise MissingFieldError(f"missing field: {key}")

def content_hash(record: dict) -> str:
    # 核心：sort_keys=True 保证乱序的字段也能算出一样的哈希
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def main():
    input_path = Path("data/dirty.jsonl")
    bad_file = Path("data/bad.jsonl")
    
    kept = 0
    dup = 0
    bad = 0
    seen_hashes = set()  # 记录见过的哈希

    # 打开 bad.jsonl 准备写入坏数据
    with bad_file.open("w", encoding="utf-8") as bf:
        for line_num, line in enumerate(load_lines(input_path), 1):
            try:
                # 1. 解析 JSON
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise BadJSONError(f"JSON parse error: {e}")

                # 2. 校验字段
                validate_record(record)

                # 3. 哈希去重
                h = content_hash(record)
                if h in seen_hashes:
                    dup += 1
                    continue
                
                seen_hashes.add(h)
                kept += 1

            except MissingFieldError as e:
                bad += 1
                # 把坏行和原因写入 bad.jsonl
                bf.write(json.dumps({"line_num": line_num, "reason": str(e), "raw": line}, ensure_ascii=False) + "\n")
            except BadJSONError as e:
                bad += 1
                bf.write(json.dumps({"line_num": line_num, "reason": str(e), "raw": line}, ensure_ascii=False) + "\n")

    # 打印汇总日志
    print(f"处理完成！")
    print(f"保留 (kept): {kept}")
    print(f"重复 (dup):  {dup}")
    print(f"坏行 (bad):  {bad}")
    print(f"详细坏行记录请看: {bad_file}")

if __name__ == "__main__":
    main()