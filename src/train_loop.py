import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import time

data_dir = Path("./data")
transform = transforms.ToTensor()
full_train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)

# 【新增 1】划分训练集和验证集 (80%训练, 20%验证)
train_size = int(0.8 * len(full_train_ds))
val_size = len(full_train_ds) - train_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) # 验证集不需要打乱

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 3
print(f"使用设备: {device} | 训练集: {train_size}张 | 验证集: {val_size}张")

for epoch in range(epochs):
    model.train() # 【新增 2】切换到训练模式（开启Dropout/BatchNorm等）
    train_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 【新增 3】打印第一个 batch 的梯度范数
        if batch_idx == 0 and epoch == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            print(f"-> 首个Batch梯度范数(L2): {total_norm ** 0.5:.4f}")
            
        optimizer.step()
        train_loss += loss.item()
        
    # 【新增 4】验证循环
    model.eval() # 切换到评估模式
    val_loss = 0.0
    correct = 0
    with torch.no_grad(): # 验证时不需要计算梯度，节省显存
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += loss_fn(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/val_size*100:.2f}%")

    # 【新增 5】保存 Checkpoint
    Path("./data/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"./data/checkpoints/model_ep{epoch}.pt")
    print(f"Epoch {epoch+1} 模型已保存。\n")