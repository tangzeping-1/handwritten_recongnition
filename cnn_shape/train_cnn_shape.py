import os
import numpy as np
from skimage.feature import hog
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ======================== 自定义数据集 ========================
class ShapeFeatureDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.cache_file = "train_features_cnn.npz" if train else "test_features_cnn.npz"
        self.expected_dim = 1296

        if os.path.exists(self.cache_file):
            print(f"\U0001F4C2 加载缓存特征：{self.cache_file}")
            with np.load(self.cache_file) as data:
                if data['features'].shape[1] != self.expected_dim:
                    print(f"⚠️ 特征维度不匹配（发现 {data['features'].shape[1]}，期望 {self.expected_dim}），重新提取...")
                    os.remove(self.cache_file)
                    self.__init__(train)
                    return
                self.features = [torch.tensor(f, dtype=torch.float32) for f in data['features']]
                self.labels = [torch.tensor(l, dtype=torch.long) for l in data['labels']]
        else:
            print(f"🔍 正在提取HOG特征（可能需要几分钟）...")
            self.data = datasets.MNIST(root='./data', train=train, download=True)
            self.features, self.labels = [], []
            for idx, (img, label) in enumerate(zip(self.data.data, self.data.targets)):
                if idx % 1000 == 0:
                    print(f"  进度：{idx}/{len(self.data.data)}")

                img_resized = Image.fromarray(img.numpy()).resize((28, 28))
                feature = hog(np.array(img_resized),
                              pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2),
                              visualize=False)
                self.features.append(torch.tensor(feature, dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.long))
            np.savez(self.cache_file,
                     features=[f.numpy() for f in self.features],
                     labels=[l.item() for l in self.labels])
            print(f"✅ 特征提取完成，已缓存到 {self.cache_file}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 重新 reshape 成伪图像 [1, 36, 36]
        reshaped = self.features[idx].view(1, 36, 36)
        return reshaped, self.labels[idx]

# ======================== CNN 模型定义 ========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [16, 18, 18]
        x = self.pool(F.relu(self.conv2(x)))  # -> [32, 9, 9]
        x = x.view(x.size(0), -1)             # 展平
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ======================== 训练流程 ========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ShapeFeatureDataset(train=True)
    test_dataset = ShapeFeatureDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"📚 Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "cnn_shape.pt")
    print("✅ 模型已保存为 cnn_shape.pt")

    # ======================== 测试准确率 ========================
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"🎯 测试准确率：{100 * correct / total:.2f}%")

# ======================== 启动 ========================
if __name__ == "__main__":
    if os.path.exists("train_features_cnn.npz"):
        os.remove("train_features_cnn.npz")

    train()