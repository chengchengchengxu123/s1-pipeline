import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 第1层卷积：输入通道1（灰度图），输出16个特征图，卷积核3x3，边缘填充1圈
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 第2层卷积：输入16，输出32，卷积核3x3，边缘填充1圈
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 最大池化层：窗口2x2，步长2（图像尺寸减半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层：此时特征图变为 32通道 x 7宽 x 7高
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x 初始形状: [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 32, 7, 7]
        
        # 展平：把多维特征图拉直，准备进全连接层
        x = x.view(x.size(0), -1)              # -> [B, 32*7*7]
        
        x = F.relu(self.fc1(x))                # -> [B, 128]
        x = self.fc2(x)                        # -> [B, 10] (输出 logits)
        return x