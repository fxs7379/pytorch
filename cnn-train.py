import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset,random_split



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 4 * 4, 128)  #图像尺寸 128x128
        self.fc2 = nn.Linear(128, 4)  # 4类:sunny, less-cloud, more-cloud, cloudy



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


        x = torch.flatten(x, 1)


        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    





root_dir='train-data'

def get_image_paths(root_dir):
    image_paths = []
    for dir_name  in ['cloudy', 'less-clouds', 'more-clouds','sunny']:
        dir_path = os.path.join(root_dir, dir_name)
        for file_name in os.listdir(dir_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dir_path, file_name))
                
    return image_paths


imagePaths=get_image_paths(root_dir)

imagePaths = sorted(list(imagePaths))


image_data = []
labels = []

for imagepath in imagePaths:
    image = cv2.imread(imagepath)
    image = cv2.resize(image,(128,128))
    image_data.append(image)

    label = imagepath.split(os.path.sep)[-2]

    labels.append(label)
    


image_data = np.array(image_data) #图像数据
labels = np.array(labels) #每个图像的标签


image_data = image_data.astype(np.float64)/255.0

dataset_size = len(image_data)
val_size = int(0.25 * dataset_size)    #用于验证
train_size = dataset_size - val_size  


label_mapping = {'cloudy': 0, 'less-clouds': 1, 'more-clouds': 2, 'sunny': 3}
labels_numeric = [label_mapping[label] for label in labels]
labels_numeric_tensor = torch.tensor(labels_numeric)

dataset = TensorDataset(torch.tensor(image_data).permute(0, 3, 1, 2).float(), labels_numeric_tensor)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 创建存储评估指标的列表
train_losses, val_accuracies, train_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], [], [], []
# 定义一个函数用于计算精确率、召回率和F1分数
def evaluate_model_advanced(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    return accuracy, precision, recall, f1






num_epochs = 200 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels_train in train_loader:
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()



    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy, _, _, _ = evaluate_model_advanced(model, train_loader)  # 计算训练集准确率
    print("train_accuracy",train_accuracy)
    train_accuracies.append(train_accuracy)
    val_accuracy, precision, recall, f1 = evaluate_model_advanced(model, val_loader)
    val_accuracies.append(val_accuracy)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


torch.save(model.state_dict(), 'model_parameters.pth')









plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(val_precisions, label='Precision', color='r')
plt.title('Validation Precision')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(val_recalls, label='Recall', color='g')
plt.title('Validation Recall')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(val_f1s, label='F1 Score', color='b')
plt.title('Validation F1 Score')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(train_accuracies, label='train_accuracies', color='b')
plt.title('Validation train_accuracies')
plt.legend()

plt.tight_layout()
plt.show()
print("已显示所有图片")