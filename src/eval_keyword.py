import json
import os
from retrieval_keyword import KeywordRetriever

# 使用配置 B (1000字符/150重叠) 的切块数据
CHUNK_FILE = "data/processed/chunks_1000_150.jsonl"
DEV_QUESTION_FILE = "data/dev/questions.jsonl"
OUTPUT_DIR = "artifacts/S3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate():
    retriever = KeywordRetriever(CHUNK_FILE)
    
    hit_count = 0
    valid_question_count = 0 # 排除那些无答案的题
    results_log = []
    
    # 读取开发集问题
    with open(DEV_QUESTION_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]
        
    for q in questions:
        qid = q["qid"]
        query = q["question"]
        gold_docs = q["gold_doc_ids"]
        answerable = q.get("answerable", True)
        
        # 调用检索器，找前 5 个块
        top_results = retriever.search(query, top_k=5)
        
        # 提取检索结果中的 doc_id
        retrieved_doc_ids = [r["doc_id"] for r in top_results]
        
        # 计算是否命中
        hit = False
        if answerable and gold_docs:
            valid_question_count += 1
            # 如果 gold_docs 里的任何一个出现在检索结果里，就算命中
            for gold in gold_docs:
                if gold in retrieved_doc_ids:
                    hit = True
                    hit_count += 1
                    break
                    
        # 记录每题情况
        results_log.append({
            "qid": qid,
            "question": query,
            "gold_docs": gold_docs,
            "retrieved_docs": retrieved_doc_ids,
            "hit_at_5": hit,
            "is_answerable": answerable
        })
        
        status = "✅命中" if hit else ("❓无答案" if not answerable else "❌漏召")
        print(f"[{qid}] {status} | 检索出: {retrieved_doc_ids}")
        
    # 计算 Recall@5
    recall = hit_count / valid_question_count if valid_question_count > 0 else 0
    print(f"\n🎯 最终 Recall@5 = {hit_count} / {valid_question_count} = {recall:.2%}")
    
    # 保存详细结果
    output_file = os.path.join(OUTPUT_DIR, "keyword_dev_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"逐题结果已保存至: {output_file}")

if __name__ == "__main__":
    evaluate()