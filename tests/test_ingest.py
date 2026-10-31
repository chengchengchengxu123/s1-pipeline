import pytest
from src.ingest import chunk_text

def test_chunk_no_overlap():
    text = "abcdefghij" # 10 个字符
    chunks = chunk_text(text, "doc_test", chunk_size=4, overlap=0)
    assert len(chunks) == 3
    assert chunks[0]["text"] == "abcd"
    assert chunks[1]["text"] == "efgh"
    assert chunks[2]["text"] == "ij"

def test_chunk_with_overlap():
    text = "abcdefghij" 
    chunks = chunk_text(text, "doc_test", chunk_size=4, overlap=2)
    assert chunks[0]["text"] == "abcd"
    assert chunks[1]["text"] == "cdef" # 往回重叠了 "cd"
    assert chunks[2]["text"] == "efgh"
    
def test_empty_text():
    chunks = chunk_text("", "doc_test", chunk_size=500)
    assert len(chunks) == 0