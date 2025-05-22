# 安装依赖
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

# 定义模型
class ANN(nn.Module):
    def __init__(self, input_dim=1296):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 画板类
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
        self.predict_button = QPushButton("识别", self)
        self.predict_button.move(100, 280)
        self.predict_button.clicked.connect(self.predict)
        self.model = ANN()
        self.model.load_state_dict(torch.load("ann_shape.pt", map_location='cpu'))
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

            img = Image.open("../ann&cnn/digit.png").convert("L").resize((28, 28))
            img_np = np.array(img)

            feature = hog(img_np, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

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
