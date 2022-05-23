import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

labels = np.loadtxt("labels.txt", dtype=np.str_)
ID_table = np.loadtxt("ID.txt", dtype=np.uint8)
label_to_id = {}
label_to_img = {}
images_list = []
id_index = 0

for i in range(len(labels)):
    image = cv2.imread("./image/" + str(i) + ".bmp")
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (77, 77))
    images_list.append(image)
    # print(image.shape[0], image.shape[1], image.shape[2])
    # os.rename("./image/" + labels[i] + ".bmp", "./image/" + str(i) + ".bmp")
    if labels[i] not in label_to_id:
        label_to_id[labels[i]] = ID_table[id_index]
        label_to_img[labels[i]] = image
        id_index += 1

data = np.array(images_list).reshape((len(images_list), -1))

# 創建一個 plot 來放我們的預期圖片
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
# 開始遍歷 plot, data, label
for ax, image, label in zip(axes, data, labels):
    # 關閉 x 軸線（不需要 x 軸線）
    ax.set_axis_off()
    # 將圖片從剛剛轉換的資料，轉回來二維的圖片
    image = image.reshape(77, 77, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 顯示圖片
    ax.imshow(image, cmap='jet', interpolation="nearest")
    # 設定標題
    ax.set_title("Training: %s" % label)

fig.savefig("output1.png")


# 創建一個 RandomForestClassifier 分類器
clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, verbose=1)

print("Fitting...")

# 訓練
clf.fit(data, labels)

image = cv2.imread("mt.png")
image_list = []
height, width, channel = image.shape

for i in range(11):
    for j in range(11):
        crop_image = image[int((i)*height/11):int((i+1)*height/11), int((j)*width/11):int((j+1)*width/11)]
        crop_image = cv2.resize(crop_image, (77, 77))
        cv2.imwrite("./analyze32/" + str(len(images_list) + 11 * i + j) + ".bmp", crop_image)
        image_list.append(crop_image)


predict_data = np.array(image_list).reshape((len(image_list), -1))

# 預測
predicted = clf.predict(predict_data)

# 用來遍歷我們的預測資料
index = 0

stageID = 39

f2 = open("Stage" + str(stageID) + "_material.txt", "w+")
f1 = open("Stage" + str(stageID) + "_entity.txt", "w+")
pref = open("predict.txt", "w+")

for i in range(11):
    for j in range(11):
        id = label_to_id[predicted[i*11+j]]
        if int(id) == 0 or int(id) == 1 or int(id) == 2 or int(id) == 73 or int(id) == 77 or int(id) == 94 or int(id) == 95:
            id = 0
        f1.write(str(id) + " ")
    f1.write("\n")

for i in range(11):
    for j in range(11):
        id = label_to_id[predicted[i*11+j]]
        if id == 0:
            f2.write(str(0) + " ")
        elif id == 2:
            f2.write(str(2) + " ")
        elif id == 73:
            f2.write(str(73) + " ")
        elif id == 77:
            f2.write(str(77) + " ")
        elif id == 94:
            f2.write(str(94) + " ")
        elif id == 95:
            f2.write(str(95) + " ")
        else:
            f2.write(str(1) + " ")
    f2.write("\n")

for i in predicted:
    pref.write(i)
    pref.write("\n")

# 創建一個 plot
fig, axes = plt.subplots(nrows=11, ncols=11, figsize=(30, 30))

# 因為有四格，所以先從欄開始
for axs in axes:

    # 再從列開始
    for ax in axs:
        # 不需要 x 軸
        ax.set_axis_off()

        # 把圖片轉回去 (28, 28) 的形式
        sk_image = predict_data[index].reshape(77, 77, 3)
        sk_image = cv2.cvtColor(sk_image, cv2.COLOR_RGB2BGR)

        # 讀入預測結果
        prediction = predicted[index]

        # 顯示圖片
        ax.imshow(sk_image, cmap=plt.cm.gray_r, interpolation="nearest")

        # 設置標題
        ax.set_title(f"Prediction: {prediction}")

        # 繼續遍歷，所以 index += 1
        index += 1

        #print('draw iamge', index)

fig.savefig("output2.png")

index = 0

fig, axes = plt.subplots(nrows=11, ncols=11, figsize=(30, 30))
# 因為有四格，所以先從欄開始
for axs in axes:

    # 再從列開始
    for ax in axs:
        # 不需要 x 軸
        ax.set_axis_off()

        # 把圖片轉回去 (28, 28) 的形式
        sk_image = label_to_img[predicted[index]]
        sk_image = cv2.cvtColor(sk_image, cv2.COLOR_RGB2BGR)

        # 讀入預測結果
        prediction = predicted[index]

        # 顯示圖片
        ax.imshow(sk_image, cmap=plt.cm.gray_r, interpolation="nearest")

        # 設置標題
        ax.set_title(f"Prediction: {prediction}")

        # 繼續遍歷，所以 index += 1
        index += 1

        #print('draw iamge', index)

fig.savefig("output3.png")

predicted = clf.predict(data)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(labels, predicted)}\n"
)