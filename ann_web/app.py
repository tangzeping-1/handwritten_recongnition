from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
import torch
import torch.nn as nn
from skimage.feature import hog

# === 模型定义 ===
class ANN(nn.Module):
    def __init__(self, input_dim=1296):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === 加载模型 ===
model = ANN()
model.load_state_dict(torch.load("ann_shape.pt", map_location="cpu"))
model.eval()

# === Flask 应用 ===
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])
    img = Image.open(io.BytesIO(image_data)).convert("L").resize((28, 28))
    img_np = np.array(img)

    feature = hog(img_np, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(feature_tensor)
        pred = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
