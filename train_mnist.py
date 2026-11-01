import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 强制使用 CPU
device = torch.device("cpu")
print(f"当前使用的计算设备: {device}")

# 数据预处理：转张量 + 标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("正在准备数据集（如果是第一次运行，会自动下载，请耐心等待）...")
# 下载并加载数据集
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

# 注意：Windows 系统下 num_workers 必须设为 0，否则容易死锁
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

# 定义简单的 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(3):
    start_time = time.time()
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | 批次 {batch_idx+1}/{len(train_loader)} | 当前 Loss: {loss.item():.4f}")

    # 每个 epoch 结束后测试准确率
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
    cost_time = time.time() - start_time
    print(f"=== Epoch {epoch+1} 完成 | 测试集准确率: {correct/total:.4f} | 耗时: {cost_time:.1f}秒 ===\n")

# 保存模型权重
torch.save(model.state_dict(), "mnist_cnn.pt")
print("训练结束！模型已保存为 mnist_cnn.pt")