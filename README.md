项目简介：基于 PyTorch 的 Fashion-MNIST 图像分类器。

环境配置：说明使用 uv 管理环境，支持 CPU-only 运行。

运行指南：uv run python -m src.train_cnn 训练，uv run python -m src.evaluate 评估。

实验结果：

TensorBoard 的训练曲线截图。

confusion_matrix.png 图片。

诊断：“模型在测试集上达到 90.x% 的准确率。通过混淆矩阵发现，模型在区分 Shirt, T-shirt, Pullover 时容易混淆，原因是分辨率过低导致纹理特征丢失。”

后续优化方向：数据增强（如 RandomHorizontalFlip）、更深的网络结构。 

S4阶段：企业级 AI Agent 核心模块

Agent 状态机：使用 LangGraph 构建 agent -> tools 循环。

双层记忆系统：LangGraph MemorySaver 维护当前会话，SQLite 实现跨会话持久化与偏好隔离。

鲁棒性与安全：实现工具异常捕获（优雅降级），并通过 System Prompt 和 ToolMessage 隔离实现防提示词注入。

## S5 阶段：评测工具 CLI
运行命令：uv run python -m src.eval_cli --input data/questions.jsonl
输出：artifacts/eval_runs/ 下的 jsonl 与 config.json。已实现 Bad JSON 拦截与单条超时隔离。
