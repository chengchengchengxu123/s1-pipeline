import os
import uuid
from pypdf import PdfReader

def read_doc(filepath):
    """读取 txt, md 或 pdf 文档，返回包含页码和文本的列表"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [{"text": f.read(), "page": 1}]
            
    elif ext == '.pdf':
        reader = PdfReader(filepath)
        pages_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages_data.append({"text": text or "", "page": i + 1})
        return pages_data
        
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def chunk_text(text, doc_id, chunk_size=500, overlap=0, page=1):
    """将长文本切分为带有重叠区的块"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk_content = text[start:end]
        chunk_id = f"{doc_id}-p{page}-{len(chunks)+1:04d}"
        
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page": page,
            "text": chunk_content,
            "start_char": start,
            "end_char": min(end, text_len)
        })
        
        if overlap == 0:
            start = end
        else:
            start = end - overlap
            
        if start >= text_len - overlap:
            break
            
    return chunks