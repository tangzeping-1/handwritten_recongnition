# app.py
from flask import Flask, render_template, request, jsonify
from cnn_model import CNN
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import base64
import numpy as np
from skimage.feature import hog

app = Flask(__name__)
model = CNN()
model.load_state_dict(torch.load("cnn_shape.pt", map_location="cpu"))
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")  # 灰度

    # Resize 到 28x28，和训练时一样
    image_resized = image.resize((28, 28))

    # 提取 HOG 特征
    feature = hog(np.array(image_resized),
                  pixels_per_cell=(4, 4),
                  cells_per_block=(2, 2),
                  visualize=False)

    # 转为 torch tensor 并 reshape 为 [1, 1, 36, 36]
    input_tensor = torch.tensor(feature, dtype=torch.float32).view(1, 1, 36, 36)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
