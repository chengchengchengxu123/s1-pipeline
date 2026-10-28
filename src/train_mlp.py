import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import time

# 1. 准备数据 (下载到项目根目录的 data 文件夹下)
data_dir = Path("./data")
transform = transforms.ToTensor()
print("正在下载/加载 FashionMNIST 数据集...")
train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
# 为了演示，我们只用前 10000 张图，跑得快一点
train_subset = torch.utils.data.Subset(train_ds, range(10000))
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

# 2. 定义模型 (多层感知机)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),              # 将 28x28 的图片展平为 784 维向量
            nn.Linear(784, 128),       # 全连接层：784 -> 128
            nn.ReLU(),                 # 激活函数
            nn.Linear(128, 10)         # 全连接层：128 -> 10 (10个类别)
        )
    def forward(self, x):
        return self.net(x)

# 3. 初始化设备与模型
# 这就是 PyTorch 最优雅的地方：自动判断是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

model = MLP().to(device)           # 把模型搬到显卡上
loss_fn = nn.CrossEntropyLoss()    # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 优化器

# 4. 训练循环
epochs = 3
print("开始训练...")
for epoch in range(epochs):
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(train_loader):
        # 【关键】把数据也搬到显卡上
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        logits = model(x)
        loss = loss_fn(logits, y)
        
        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
    print(f"Epoch {epoch+1} 完成，耗时: {time.time() - start_time:.2f} 秒\n")