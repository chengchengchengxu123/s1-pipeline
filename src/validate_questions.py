import json
import os

def validate_jsonl(filepath):
    print(f"正在校验: {filepath}")
    qids = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # 检查必填字段
                required = ["qid", "question", "gold_doc_ids", "answerable"]
                for req in required:
                    if req not in data:
                        raise ValueError(f"第 {line_num} 行缺少字段: {req}")
                
                # 检查 qid 是否重复
                if data["qid"] in qids:
                    raise ValueError(f"第 {line_num} 行发现重复的 qid: {data['qid']}")
                qids.add(data["qid"])
                
            except json.JSONDecodeError:
                print(f"❌ 第 {line_num} 行不是合法的 JSON: {line.strip()}")
            except Exception as e:
                print(f"❌ 校验失败: {e}")

    print(f"✅ {filepath} 校验完成，共 {len(qids)} 题。")

if __name__ == "__main__":
    validate_jsonl("data/dev/questions.jsonl")
    validate_jsonl("data/test/questions.jsonl")