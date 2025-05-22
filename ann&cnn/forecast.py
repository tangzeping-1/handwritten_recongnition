import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk

# 选择模型类型（ANN 或 CNN）
from ann_model import ANN

# 初始化模型
model = ANN()  # 若使用 CNN，请替换为 CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("ann_mnist.pt", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)

def launch_drawing_interface(model):
    width, height = 280, 280  # 放大画布
    image1 = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image1)

    # 创建窗口
    window = tk.Tk()
    window.title("手写数字识别")

    canvas = tk.Canvas(window, width=width, height=height, bg='white')
    canvas.pack()

    def paint(event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        canvas.create_oval(x1, y1, x2, y2, fill='black')
        draw.ellipse([x1, y1, x2, y2], fill=0)

    canvas.bind("<B1-Motion>", paint)

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, width, height], fill=255)
        result_label.config(text="绘制数字后点击预测")

    def predict():
        img_resized = image1.resize((28, 28))
        img_tensor = transforms.ToTensor()(img_resized)
        img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)
            result_label.config(text=f"预测结果：{pred.item()}")

    # 按钮与结果
    btn_frame = tk.Frame(window)
    btn_frame.pack()
    tk.Button(btn_frame, text="预测", command=predict).pack(side=tk.LEFT)
    tk.Button(btn_frame, text="清空", command=clear).pack(side=tk.LEFT)
    result_label = tk.Label(window, text="绘制数字后点击预测", font=('Arial', 16))
    result_label.pack()

    window.mainloop()

# 启动绘图预测
launch_drawing_interface(model)
