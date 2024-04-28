
import cv2
from flask import Flask, request, jsonify
import torch.nn.functional as F
from joblib import load
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)


        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 4 * 4, 128)  # Assuming input image size is 128x128
        self.fc2 = nn.Linear(128, 4)  # 4 classes: sunny, less-cloud, more-cloud, cloudy

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv6(x)
        # x = F.relu(x)
        # x = self.pool(x)

        x = torch.flatten(x, 1)

        # x = x.view(-1, 256 * 4* 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

# 步骤1: 定义 Flask 应用
app = Flask(__name__)

knn = load("knn_5_model.joblib")
rf = load("random_forest_model.joblib")




model = CNN()
state_dict = torch.load('model_parameters.pth')
model.load_state_dict(state_dict) 

model.eval()
@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    # 从请求中获取图像文件
    file = request.files['image']
    # 将图像文件转换为 PIL 图像对象
    image = Image.open(file)
    # 对图像进行预处理
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小
        transforms.ToTensor(),         # 转换为张量
    ])
    image = preprocess(image)
    # 增加批次维度
    image = image.unsqueeze(0)
    # 使用模型进行预测
    with torch.no_grad():  # 禁用梯度计算，因为我们只是在做推理
        output = model(image)
    prediction = output.argmax().item()  # 获取最有可能的类别
    return jsonify({'prediction': prediction})



@app.route('/predict_knn',methods = ['post'])
def predict_knn():
    try:
        # 检查是否有文件且是否为图片
        file = request.files['image']
        # if file and 'image' in file.content_type:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 确保图像被调整到与模型训练时相同的尺寸
        image = cv2.resize(image, (128, 128))
        image = image.reshape(1, -1)
        image = image / 255.0  # 归一化

        prediction = knn.predict(image)
        return jsonify({'prediction': int(prediction[0])})
        # else:
        #     return jsonify({'error': 'No image or incorrect file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/predict_rf',methods = ['post'])
def predict_rf():
    try:
        # 检查是否有文件且是否为图片
        file = request.files['image']
        # if file and 'image' in file.content_type:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


        image = cv2.resize(image, (128, 128))
        image = image.reshape(1, -1)
        image = image / 255.0  # 归一化

        prediction = rf.predict(image)
        return jsonify({'prediction': int(prediction[0])})
        # else:
        #     return jsonify({'error': 'No image or incorrect file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500






# 启动应用
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8082)