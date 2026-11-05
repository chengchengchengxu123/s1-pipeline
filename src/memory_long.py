import os
import json, ast, operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ================= 1. 定义工具 =================
@tool
def search_docs(query: str) -> str:
    """检索项目文档。当用户询问具体知识或概念时使用此工具。"""
    return f"【模拟检索结果】关于 '{query}' 的相关文档：Transformer 的核心是自注意力机制..."

@tool
def calculate(expression: str) -> str:
    """数学计算。用于计算加减乘除、乘方等数学表达式。参数示例：(12+8)*3"""
    return "20"

@tool
def query_metadata(name: str) -> str:
    """查询数据文件的统计信息（如记录条数）。参数 name 目前只支持 'questions'。"""
    try:
        path = "data/test/questions.jsonl"
        count = 0
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip(): count += 1
        return f"questions.jsonl 中共有 {count} 条记录"
    except Exception as e:
        return f"[工具异常] 元数据查询失败：{e}"

tools = [search_docs, calculate, query_metadata]

# ================= 2. 初始化模型 =================
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash", 
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ================= 3. 定义状态 =================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ================= 4. 定义节点逻辑 =================
def call_model(state: AgentState):
    # 1. 获取当前用户 ID (实际开发中通常从 state 或 config 传入)
    user_id = "user_001" 
    
    # 2. 从 SQLite 查库，获取长期记忆
    facts = get_facts(user_id)
    memory_prompt = ""
    if facts:
        facts_str = "\n".join([f"- 用户的 {k} 是: {v}" for k, v in facts.items()])
        memory_prompt = f"\n\n【用户的长期偏好档案】：\n{facts_str}\n请在回答时结合这些偏好。"
    
    # 3. 把长期记忆注入到 SystemMessage
    system_msg = SystemMessage(content=f"你是一个智能助手。{memory_prompt}")
    
    messages = state["messages"]
    
    # 4. 将 SystemMessage 和最近的消息拼在一起
    # （为了简化，这里就不写滑动窗口了，直接拼在历史消息最前面）
    final_messages = [system_msg] + list(messages)
    
    # 5. 请求模型
    response = llm_with_tools.invoke(final_messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ================= 5. 构建图 =================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# 核心修复：在这里直接挂载 MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


    



import sqlite3
import time

DB_PATH = "data/memory.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 允许以字典形式访问列
    return conn

def save_fact(user_id: str, key: str, value: str):
    """保存或更新用户偏好（解决冲突覆盖）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_facts (user_id, fact_key, fact_value, timestamp, status)
        VALUES (?, ?, ?, ?, 'active')
    ''', (user_id, key, value, int(time.time())))
    conn.commit()
    conn.close()

def get_facts(user_id: str):
    """获取用户所有活跃的偏好"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT fact_key, fact_value FROM user_facts WHERE user_id = ? AND status = "active"', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    # 转换为字典格式 {"preferred_language": "Python"}
    return {row["fact_key"]: row["fact_value"] for row in rows}

# 增加一个提取节点
def extract_memory(state: AgentState):
    # 拿到最新的两句话（用户提问 + 模型回答）
    messages = state["messages"]
    if len(messages) < 2:
        return state
    
    last_user_msg = messages[-2].content
    
    # 用 Prompt 让模型判断是否要提取
    extract_prompt = f"""
    请分析用户的话是否包含长期偏好（如喜好、习惯、个人信息等）。
    如果有，请输出 JSON：{{"key": "偏好的类别", "value": "偏好的内容"}}
    如果没有，请输出 {{"key": None, "value": None}}。
    用户的话："{last_user_msg}"
    """
    # 这里为了简单，不用工具，直接调 llm
    res = llm.invoke([HumanMessage(content=extract_prompt)])
    
    try:
        # 解析模型的 JSON 返回
        import json
        data = json.loads(res.content)
        if data.get("key"):
            # 写入 SQLite
            save_fact("user_001", data["key"], data["value"])
            print(f">>> [后台提取] 成功提取并保存长期记忆：{data['key']} = {data['value']}")
    except:
        pass
    
    return state

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # 先清空之前的测试数据，保证干净开始
    conn = get_db_connection()
    conn.execute("DELETE FROM user_facts WHERE user_id = 'user_001'")
    conn.commit()
    conn.close()

    # 封装一个通用的提问函数，省得每次都写 for 循环
    def ask_agent(question, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        input_msg = {"messages": [HumanMessage(content=question)]}
        ans = ""
        for event in app.stream(input_msg, config=config):
            for key, value in event.items():
                if key == "agent":
                    ans = value['messages'][-1].content
        return ans

    print("="*40)
    print("🧪 测试场景 1：记忆的【冲突覆盖】")
    print("="*40)
    # 1. 写入偏好：Python
    save_fact("user_001", "编程语言", "Python")
    print("存入偏好：Python")
    
    # 2. 直接覆盖为：Java
    print(">>> 用户改变主意了，现在覆盖为：Java")
    save_fact("user_001", "编程语言", "Java")
    
    # 3. 在全新会话中问模型
    q1 = "我该用什么语言写代码比较好？"
    print(f"\n用户提问: {q1}")
    print(f"Agent回答: {ask_agent(q1, 'test_session_1')}")
    # 预期：模型必须回答 Java，说明旧数据被完美覆盖，不会出现两个偏好打架。


    print("\n" + "="*40)
    print("🧪 测试场景 2：记忆的【过期删除（软删除）】")
    print("="*40)
    # 1. 将该用户的记忆标记为过期
    conn = get_db_connection()
    conn.execute("UPDATE user_facts SET status = 'expired' WHERE user_id = 'user_001'")
    conn.commit()
    conn.close()
    print(">>> 已将该用户的记忆状态改为 'expired'（过期）")

    # 2. 再次在全新会话中问同样的问题
    q2 = "我该用什么语言写代码比较好？"
    print(f"\n用户提问: {q2}")
    print(f"Agent回答: {ask_agent(q2, 'test_session_2')}")
    # 预期：因为查不到 active 状态的记忆，Agent 不会提 Java，而是给出通用建议。


    print("\n" + "="*40)
    print("🧪 测试场景 3：记忆的【彻底抹除（用户注销）】")
    print("="*40)
    # 1. 直接物理删除该用户的所有记录
    conn = get_db_connection()
    conn.execute("DELETE FROM user_facts WHERE user_id = 'user_001'")
    conn.commit()
    conn.close()
    print(">>> 已执行物理删除：DELETE FROM user_facts")
    
    # 2. 验证数据库是否真的空了
    remaining = get_facts("user_001")
    print(f"当前数据库里 user_001 的剩余记录: {remaining}")
    # 预期：输出为空字典 {}，证明数据被彻底灰飞烟灭，符合数据隐私保护要求。