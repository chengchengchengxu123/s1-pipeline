项目简介：基于 PyTorch 的 Fashion-MNIST 图像分类器。

环境配置：说明使用 uv 管理环境，支持 CPU-only 运行。

运行指南：uv run python -m src.train_cnn 训练，uv run python -m src.evaluate 评估。

实验结果：

贴上 TensorBoard 的训练曲线截图。

贴上刚生成的 confusion_matrix.png 图片。

用一两句话写出你的诊断：“模型在测试集上达到 90.x% 的准确率。通过混淆矩阵发现，模型在区分 Shirt, T-shirt, Pullover 时容易混淆，原因是分辨率过低导致纹理特征丢失。”

后续优化方向：数据增强（如 RandomHorizontalFlip）、更深的网络结构。 