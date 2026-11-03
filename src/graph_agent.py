from langchain_core.tools import tool
import json, ast, operator
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search_docs(query: str) -> str:
    """检索项目文档。当用户询问具体知识或概念时使用此工具。"""
    # 复制昨天的模拟代码或真实 BM25 代码
    return f"【模拟检索结果】关于 '{query}' 的相关文档：Transformer 的核心是自注意力机制..."

@tool
def calculate(expression: str) -> str:
    """数学计算。用于计算加减乘除、乘方等数学表达式。参数示例：(12+8)*3"""
    # 复制昨天的 ast 安全计算代码
    # ... 省略内部实现 ...
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


import os
from langchain_openai import ChatOpenAI

# 注意：这里要填你的真实 Key
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash", # 用昨天开通的免费模型
    temperature=0
)

# 关键一步：把工具绑定到模型上
llm_with_tools = llm.bind_tools(tools)


from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 消息列表，add_messages 负责把新消息追加进去
    messages: Annotated[Sequence[BaseMessage], add_messages]

    from langgraph.prebuilt import ToolNode

def call_model(state: AgentState):
    messages = state["messages"]
    # 模型会根据绑定的工具，决定是输出回答，还是输出 tool_calls
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 框架内置的工具执行节点
tool_node = ToolNode(tools)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

# 1. 初始化图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 3. 设置入口边
workflow.add_edge(START, "agent")

# 4. 设置条件路由（划重点！）
# tools_condition 是框架内置函数：
# 它会检查 agent 节点输出的消息里有没有 tool_calls。
# 如果有，路由去 "tools" 节点；如果没有（输出最终文本了），路由去 END。
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
# 5. 工具执行完后，永远回到 agent 节点，形成闭环
workflow.add_edge("tools", "agent")

# 6. 编译成可执行的应用
app = workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    query = "项目里 questions.jsonl 有多少条记录？"
    print(f"用户: {query}")
    
    # 初始状态只有一条用户消息
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # 运行图 (stream模式可以看到每一步的流转)
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\n--- 节点: {key} ---")
            # 打印该节点产生的最新消息
            print(value["messages"][-1])