import os
import json
from ingest import read_doc, chunk_text

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

def process_all(chunk_size, overlap, output_filename):
    all_chunks = []
    
    for filename in os.listdir(RAW_DIR):
        if not filename.startswith("doc_"): continue
        
        filepath = os.path.join(RAW_DIR, filename)
        doc_id = os.path.splitext(filename)[0]
        
        try:
            # 👇 就是漏了这一行，读取文件内容
            pages = read_doc(filepath) 
            
            for p in pages:
                if not p["text"].strip():
                    print(f"⚠️ 警告: {doc_id} 第 {p['page']} 页没有提取到文本")
                    continue
                    
                chunks = chunk_text(p["text"], doc_id, chunk_size, overlap, p["page"])
                all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ 解析 {filename} 失败: {e}")
            
    output_path = os.path.join(OUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    print(f"✅ 处理完成，共生成 {len(all_chunks)} 个块 -> {output_path}")

if __name__ == "__main__":
    print("=== 实验 A: chunk_size=500, overlap=0 ===")
    process_all(500, 0, "chunks_500_0.jsonl")
    
    print("\n=== 实验 B: chunk_size=1000, overlap=150 ===")
    process_all(1000, 150, "chunks_1000_150.jsonl")