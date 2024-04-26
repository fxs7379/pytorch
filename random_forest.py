
from joblib import dump
import numpy as np
import cv2
import os
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support



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
label_mapping = {'cloudy': 0, 'less-clouds': 1, 'more-clouds': 2, 'sunny': 3}
labels_numeric = [label_mapping[label] for label in labels]
n_samples = len(image_data)
data = image_data.reshape((n_samples, -1))  # 将每个图像转为一维数组
data = data / 255.0
print(len(data))
print(data)


X_train, X_test, y_train, y_test = train_test_split(data, labels_numeric, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

dump(clf,"random_forest_model.joblib")

y_pred = clf.predict(X_test)



print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 计算各项指标
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)

# 绘制性能指标图表
labels = np.unique(y_test)
x = np.arange(len(labels))  # 标签位置
width = 0.2  # 柱状图的宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, fscore, width, label='F1 Score')

# 添加一些文本标签、标题和自定义x轴标签等
ax.set_ylabel('Scores')
ax.set_title('Scores by group and metric')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
