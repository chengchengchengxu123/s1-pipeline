import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import copy

data_dir = Path("./data")
transform = transforms.ToTensor()
# 1. 故意制造小数据：只取前 2000 张图
full_train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
small_ds = Subset(full_train_ds, range(2000))

# 划分 1500 训练，500 验证
train_size = int(0.75 * 2000)
val_size = 2000 - train_size
train_ds, val_ds = random_split(small_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 2. 故意放大模型：增加神经元数量
class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(784, 512), nn.ReLU(), # 原来是 128
            nn.Linear(512, 256), nn.ReLU(), # 增加了一层
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

# 3. 定义训练函数
def run_experiment(lr, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LargeMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        t_loss, t_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_correct += (logits.argmax(1) == y).sum().item()
            
        # 验证阶段
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += loss_fn(logits, y).item()
                v_correct += (logits.argmax(1) == y).sum().item()
                
        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["train_acc"].append(t_correct / train_size)
        history["val_acc"].append(v_correct / val_size)
        
    return history

# 4. 运行两组不同学习率的实验
print("开始运行 LR=0.001 实验...")
hist_lr1 = run_experiment(lr=1e-3)
print("开始运行 LR=0.01 实验...")
hist_lr2 = run_experiment(lr=1e-2)

# 5. 绘制对比图
epochs = range(1, 31)
plt.figure(figsize=(12, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, hist_lr1["train_loss"], 'b--', label='Train Loss (lr=1e-3)')
plt.plot(epochs, hist_lr1["val_loss"], 'b-', label='Val Loss (lr=1e-3)')
plt.plot(epochs, hist_lr2["train_loss"], 'r--', label='Train Loss (lr=1e-2)')
plt.plot(epochs, hist_lr2["val_loss"], 'r-', label='Val Loss (lr=1e-2)')
plt.title("Overfitting & Learning Rate Comparison (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, hist_lr1["train_acc"], 'b--', label='Train Acc (lr=1e-3)')
plt.plot(epochs, hist_lr1["val_acc"], 'b-', label='Val Acc (lr=1e-3)')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("overfit_curve.png")
print("实验完成，图表已保存为 overfit_curve.png")