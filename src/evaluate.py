import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from src.dataset import get_fashion_mnist_loaders
from src.models import FashionCNN

def evaluate():
    device = torch.device("cpu")
    
    # 1. 加载测试集数据
    _, test_loader = get_fashion_mnist_loaders(batch_size=256)
    
    # 2. 实例化模型并加载昨天保存的权重
    model = FashionCNN().to(device)
    weight_path = 'artifacts/fashion_cnn.pth'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到模型文件: {weight_path}，请先运行 train_cnn.py")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"✅ 成功加载模型权重: {weight_path}")

    # 3. 切换到评估模式 (极其重要！关闭 Dropout 和 BatchNorm 的训练行为)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 4. 开始在测试集上推理
    with torch.no_grad(): # 极其重要！关闭梯度计算，节省内存并加速
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 取出概率最大的类别索引
            _, predicted = torch.max(outputs, 1)
            
            # 将 Tensor 转回 CPU 并转为 numpy 数组，存入列表
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 5. 计算测试集准确率
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    
    # 6. 绘制混淆矩阵
    print("📊 正在绘制混淆矩阵...")
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
                   
    # 使用行归一化，让对角线显示为召回率 (0-1之间)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', values_format='.2f', ax=ax, xticks_rotation='vertical')
    
    plt.title(f"Fashion-MNIST Confusion Matrix (Test Acc: {test_acc:.2f})")
    plt.tight_layout()
    
    # 7. 保存图片到 artifacts 目录
    os.makedirs('artifacts', exist_ok=True)
    img_path = 'artifacts/confusion_matrix.png'
    plt.savefig(img_path, dpi=200)
    print(f"✅ 混淆矩阵已保存至: {img_path}")

if __name__ == "__main__":
    evaluate()