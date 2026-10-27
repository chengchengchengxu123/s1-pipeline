from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
# 复用昨天写好的哈希函数
from src.clean import content_hash

app = FastAPI(title="S1 数据清洗 API")

# 1. 定义数据库文件路径
DB_PATH = Path("data/s1_database.db")

# 2. 定义 Pydantic 模型，用于自动校验请求体
class IngestItem(BaseModel):
    doc_id: str
    source: str
    content: str

# 3. 初始化数据库表
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# 启动服务时会自动建表
init_db()

# --- 路由部分 ---

# GET /health
@app.get("/health")
def health_check():
    return {"status": "ok"}

# POST /ingest
@app.post("/ingest", status_code=201)
def ingest_data(item: IngestItem):
    # 1. 计算哈希（复用 clean.py 里的逻辑）
    record_dict = item.model_dump()
    h = content_hash(record_dict)
    
    # 2. 获取当前时间
    now = datetime.now(timezone.utc).isoformat()
    
    # 3. 写入 SQLite 数据库
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 参数化查询，防止 SQL 注入（注意这里的 ? 占位符）
        cursor.execute(
            "INSERT INTO documents (doc_id, source, content_hash, ingested_at) VALUES (?, ?, ?, ?)",
            (item.doc_id, item.source, h, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库写入失败: {str(e)}")

    return {
        "message": "数据入库成功",
        "doc_id": item.doc_id,
        "hash": h,
        "ingested_at": now
    }