import torch

def check_autograd():
    # 1. 模拟模型参数 (requires_grad=True)
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # 2. 模拟前向传播
    x = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) # 假输入
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # 假目标
    
    y_pred = weights @ x
    loss = ((y_pred - y_true) ** 2).mean()
    
    print(f"Loss: {loss.item():.4f}")
    
    # 3. 反向传播
    loss.backward()
    
    # 4. 查看梯度
    print("Gradients:\n", weights.grad)
    print("Is leaf:", weights.is_leaf)

if __name__ == "__main__":
    check_autograd()