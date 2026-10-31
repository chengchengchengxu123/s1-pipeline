import json
import jieba
from rank_bm25 import BM25Okapi

class KeywordRetriever:
    def __init__(self, chunk_filepath):
        self.chunks = []
        self.tokenized_corpus = []
        
        # 1. 加载昨天切好的数据（我们选 1000 字大小的块，上下文更完整）
        with open(chunk_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
                    
        # 2. 对所有文本进行中文分词
        print(f"正在对 {len(self.chunks)} 个块进行分词...")
        for chunk in self.chunks:
            # jieba.lcut 返回一个列表，比如 ["PyTorch", "是", "一个", "框架"]
            tokens = list(jieba.cut(chunk["text"]))
            self.tokenized_corpus.append(tokens)
            
        # 3. 构建 BM25 索引
        print("正在构建 BM25 索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("索引构建完成！")

    def search(self, query, top_k=5):
        """输入问题，返回 top_k 个相关的证据块"""
        # 对用户的提问也进行同样的分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取所有块的 BM25 分数
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 排序并取前 top_k 个
        # argsort 返回的是从小到大的索引，[::-1] 反转变成从大到小
        top_indices = doc_scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = doc_scores[idx]
            # 如果分数为 0，说明连一个关键词都没匹配上，直接丢弃
            if score > 0:
                results.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "doc_id": self.chunks[idx]["doc_id"],
                    "score": float(score),
                    "text": self.chunks[idx]["text"]
                })
        return results