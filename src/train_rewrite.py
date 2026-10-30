import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 准备数据 (FashionMNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transform)
val_dataset = datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 2. 定义模型 (一个简单的多层感知机 MLP)
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# 3. 初始化模型、损失函数和优化器
device = torch.device("cpu")
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 核心训练循环
epochs = 3 
train_losses = []
val_losses = []

for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()  # 切换到训练模式（启用 Dropout 和 BatchNorm）
    running_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # === 补全的 4 步核心逻辑 ===
        # 1. 梯度清零 (如果不清零，梯度会累加)
        optimizer.zero_grad()
        
        # 2. 前向传播 (把图片输入模型，得到预测结果)
        outputs = model(batch_x)
        
        # 3. 计算损失 (对比预测结果和真实标签)
        loss = criterion(outputs, batch_y)
        
        # 4. 反向传播 (计算梯度) + 更新参数
        loss.backward()
        optimizer.step()
        # =========================
        
        running_loss += loss.item()
        
    train_losses.append(running_loss / len(train_loader))
    
    # --- 验证阶段 ---
    model.eval()  # 切换到评估模式（关闭 Dropout 和 BatchNorm）
    val_loss = 0.0
    
    # 验证阶段不需要计算梯度，用 torch.no_grad() 节省内存和计算资源
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
    val_losses.append(val_loss / len(val_loader))
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

# 5. 画出 Loss 曲线并保存
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("loss_curve_rewrite.png")
print("曲线图已保存为 loss_curve_rewrite.png")