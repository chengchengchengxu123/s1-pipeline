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
    messages = state["messages"]
    
    # 进阶：如果消息超过 10 条，触发摘要压缩（昨天写的逻辑）
    if len(messages) > 10:
        old_messages = messages[:4]
        recent_messages = messages[4:]
        
        summary_prompt = f"请用一句话总结以下对话的核心内容（特别是用户的偏好或事实）：\n{old_messages}"
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        
        summary_msg = SystemMessage(content=f"【历史对话摘要】：{summary_response.content}")
        final_messages = [summary_msg] + recent_messages
    else:
        final_messages = messages

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

# ================= 6. 测试代码 =================
if __name__ == "__main__":
    # 配置 thread_id 进行会话隔离（非常关键！）
    config = {"configurable": {"thread_id": "user_001_session_01"}}
    
    print("\n--- 第一轮 ---")
    print("用户: 我叫小明，请记住。")
    input_1 = {"messages": [HumanMessage(content="我叫小明，请记住。")]}
    # 必须把 config 传给 stream
    for event in app.stream(input_1, config=config):
        pass 
    
    print("\n--- 第二轮 ---")
    print("用户: 我叫什么名字？")
    input_2 = {"messages": [HumanMessage(content="我叫什么名字？")]}
    # 再次传入一模一样的 config
    for event in app.stream(input_2, config=config):
        for key, value in event.items():
            if key == "agent":
                print(f"Agent: {value['messages'][-1].content}")
    print("\n--- 换一个用户 (thread_id 不同) ---")
    config_2 = {"configurable": {"thread_id": "user_002_session_01"}}
    print("用户 (新用户): 我叫什么名字？")
    input_3 = {"messages": [HumanMessage(content="我叫什么名字？")]}
    for event in app.stream(input_3, config=config_2):
        for key, value in event.items():
            if key == "agent":
                print(f"Agent: {value['messages'][-1].content}")
    

