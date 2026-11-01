import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from src.dataset import get_fashion_mnist_loaders
from src.models import FashionCNN

def train():
    # 1. 环境与设备配置 (目前强制使用 CPU 规避版本冲突)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. 准备数据
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64)
    
    # 3. 初始化模型、损失函数、优化器
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器，学习率设为 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 初始化 TensorBoard 的 Writer (日志会保存在 runs/ 目录下)
    writer = SummaryWriter('runs/fashion_cnn_experiment_1')
    
    epochs = 10 # 设定训练 10 轮
    
    # 5. 开始训练循环
    for epoch in range(epochs):
        model.train() # 切换到训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 使用 tqdm 包装 DataLoader，显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # --- 核心五步 ---
            optimizer.zero_grad()          # 1. 清空梯度
            outputs = model(images)        # 2. 前向传播
            loss = criterion(outputs, labels) # 3. 计算损失
            loss.backward()                # 4. 反向传播 (Autograd)
            optimizer.step()               # 5. 更新参数
            
            # 统计指标
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 实时更新进度条上的 Loss
            pbar.set_postfix({'loss': loss.item()})
            
        # 计算当前 Epoch 的平均 Loss 和 训练准确率
        epoch_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        print(f"Epoch {epoch+1} Summary: Train Loss = {epoch_loss:.4f}, Train Acc = {train_acc:.4f}")
        
    # 6. 训练结束，保存模型权重 (存到 artifacts 目录)
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), 'artifacts/fashion_cnn.pth')
    print("Model saved to artifacts/fashion_cnn.pth")
    
    writer.close()

if __name__ == "__main__":
    train()