# pip install PyQt5 Pillow scikit-image torch torchvision

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from skimage.feature import hog
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN 模型定义（必须和训练时一致）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [16,18,18]
        x = self.pool(F.relu(self.conv2(x)))  # -> [32,9,9]
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 主窗口类
class DrawBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('手绘数字识别')
        self.setFixedSize(280, 300)
        self.canvas = QPixmap(280, 280)
        self.canvas.fill(Qt.white)
        self.label = QLabel(self)
        self.label.setPixmap(self.canvas)
        self.label.move(0, 0)
        self.last_point = QPoint()

        # 按钮
        self.predict_button = QPushButton("识别", self)
        self.predict_button.move(100, 280)
        self.predict_button.clicked.connect(self.predict)

        # 模型加载
        self.model = CNN()
        self.model.load_state_dict(torch.load("cnn_shape.pt", map_location='cpu'))
        self.model.eval()

    def mousePressEvent(self, event):
        self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        painter = QPainter(self.label.pixmap())
        pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self.last_point, event.pos())
        self.last_point = event.pos()
        self.update()

    def predict(self):
        print("🔍 开始识别")
        try:
            saved = self.label.pixmap().save("digit.png")
            if not saved:
                print("❌ 保存图像失败")
                return
            print("✅ 图像已保存")

            # 加载图像并提取 HOG 特征
            img = Image.open("../ann&cnn/digit.png").convert("L").resize((28, 28))
            img_np = np.array(img)
            feature = hog(img_np, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)

            # 转为 CNN 输入格式：1×1×36×36
            feature_tensor = torch.tensor(feature, dtype=torch.float32).reshape(1, 1, 36, 36)

            with torch.no_grad():
                output = self.model(feature_tensor)
                pred = torch.argmax(output, dim=1).item()

            self.setWindowTitle(f"识别结果：{pred}")
            print(f"🎯 识别结果：{pred}")

        except Exception as e:
            print(f"❌ 出错：{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrawBoard()
    window.show()
    sys.exit(app.exec_())
