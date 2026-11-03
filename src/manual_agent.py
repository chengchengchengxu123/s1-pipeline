import json
import ast
import operator
from src.provider import get_provider
# 假设你 retrieval_keyword.py 里的类名是 BM25Retriever，请根据实际类名修改
from src.retrieval_keyword import KeywordRetriever 

# 1. 搜索工具
# 注意：如果你的 retriever 需要提前 ingest 数据，请确保在这里或外部已经完成了初始化
#retriever = KeywordRetriever(chunk_filepath="data/chunks/processed_chunks.jsonl")  # 如果类名不一样，请修改

# 注释掉真实的检索器实例化
# retriever = KeywordRetriever(chunk_filepath="data/chunks/processed_chunks.jsonl")

def search_docs(query: str) -> str:
    # 临时返回假数据，先确保 Agent 循环能走通
    print(f"   [工具执行] 正在检索：{query}")
    return f"【模拟检索结果】关于 '{query}' 的相关文档：Transformer 的核心是自注意力机制，它允许模型关注输入序列的不同位置。"

    # 真实的检索代码先注释掉
    # try:
    #     results = retriever.search(query, top_k=3)
    #     ...

# 2. 计算工具
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg}

def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("不支持的表达式")

def calculate(expression: str) -> str:
    try:
        return str(_eval_node(ast.parse(expression, mode="eval")))
    except Exception as e:
        return f"[工具异常] 计算失败：{e}"

# 3. 元数据查询工具（根据截图3读取 questions.json）
def query_metadata(name: str) -> str:
    try:
        if name == "questions":
            # 暂时不读文件了，直接返回假数据，先测试真实大模型的调用能力
            return "questions.json 中共有 10 条记录（模拟数据）"
        return f"未知元数据：{name}"
    except Exception as e:
        return f"[工具异常] 元数据查询失败：{e}"

# 工具白名单
TOOLS = {
    "search_docs": search_docs,
    "calculate": calculate,
    "query_metadata": query_metadata,
}
SYSTEM_PROMPT = """你是一个可以使用工具的 Agent。你只能输出一个 JSON 对象，不要输出 Markdown 代码块或额外解释。

可用工具：
1. search_docs：检索项目文档。参数：{"query": "检索词"}
2. calculate：数学计算。参数：{"expression": "如 (12+8)*3"}
3. query_metadata：查询元数据。参数：{"name": "questions"}

输出格式二选一：
A. 调用工具：{"action": "工具名", "args": {参数对象}}
B. 最终回答：{"action": "final_answer", "answer": "给用户的答案"}

规则：
- 每次只调用一个工具。
- 如果已有信息足够回答，输出 final_answer。
- 不要编造不存在的工具。
"""



def run_manual_agent(user_question: str, max_steps: int = 5):
    # 1. 初始化 Provider (先用 Mock 测试逻辑，跑通后改成 use_mock=False)
    provider = get_provider(use_mock=False)  
    
    # 2. 拼接初始 Prompt
    prompt = f"{SYSTEM_PROMPT}\n\n用户问题：{user_question}\n"
    
    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")
        
        # 3. 调用模型，要求返回 JSON
        # 你的 provider 中有 expect_json=True，它会尝试解析并返回字典
        try:
            action_obj = provider.chat(prompt, expect_json=True)
        except Exception as e:
            print(f"模型返回解析失败: {e}")
            break
            
        print(f"模型决策: {action_obj}")
        
        action = action_obj.get("action")
        
        # 4. 判断是否为最终回答
        if action == "final_answer":
            print("\n=== Final Answer ===")
            print(action_obj.get("answer"))
            return action_obj.get("answer")
            
        # 5. 白名单校验
        if action not in TOOLS:
            obs = f"错误：工具 {action} 不存在。可用工具：{list(TOOLS.keys())}。请重新选择。"
            print(f"Observation: {obs}")
            # 将错误拼回 prompt，让模型下一轮修正
            prompt += f"\n你的回答：{json.dumps(action_obj, ensure_ascii=False)}\nObservation: {obs}\n"
            continue
            
        # 6. 执行工具
        args = action_obj.get("args", {})
        print(f"执行工具: {action}, 参数: {args}")
        try:
            result = TOOLS[action](**args)
        except Exception as e:
            result = f"[工具异常] {action} 执行失败：{e}"
            
        obs = str(result)[:800]
        print(f"Observation: {obs[:200]}...") # 终端只打印前200字符
        
        # 7. 将结果拼回 prompt，进入下一轮
        prompt += f"\n你的回答：{json.dumps(action_obj, ensure_ascii=False)}\nObservation: {obs}\n请根据 Observation 继续输出 JSON。"

    print("\n达到最大步数，强制终止。")
    return "达到最大步数，未能得到最终答案。"

if __name__ == "__main__":
    # 测试用例
    test_q = "项目里 questions.json 有多少条记录？"
    run_manual_agent(test_q)