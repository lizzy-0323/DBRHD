# coding: utf-8
"""
@author: laziyu
@date: 2023/11/13
@desc: using multiple method to recognize the handwritten digits
"""
import os
import random

import numpy as np
from knn import KNN
from net_naive import NET
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import train_test_split as TLS


def load_data(path="./data", train=True):
    if os.path.exists(path):
        # 读取训练数据
        if train:
            data_path = os.path.join(path, "trainingDigits")
        # 读取测试数据
        else:
            data_path = os.path.join(path, "testDigits")
        file_list = os.listdir(data_path)
        file_num = len(file_list)
        dataset = np.zeros((file_num, 1024))
        labels = np.zeros((file_num, 1))
        for i in range(file_num):
            file_path = file_list[i]
            file = os.path.join(data_path, file_path)
            label = int(file_path.split("_")[0])
            labels[i] = label
            with open(file, "r", encoding="utf-8") as f:
                for j in range(32):
                    line = f.readline()
                    for k in range(32):
                        dataset[i, 32 * j + k] = int(line[k])
        return dataset, labels

    print("The path is not exist!")


def run(method="knn"):
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    acc = 0
    if method == "knn":
        knn = KNN(train_data, train_labels, k=3)
        acc = knn.fit(test_data, test_labels, k=3)
    elif method == "net":
        train_data, valid_data, train_labels, valid_labels = TLS(
            train_data, train_labels, test_size=0.2, random_state=420
        )
        dnn = DNN(hidden_layer_sizes=(100,), random_state=420).fit(
            train_data, train_labels.ravel()
        )
        # 采用交叉验证
        # print(dnn.score(valid_data, valid_labels))
        acc = dnn.score(test_data, test_labels)
    elif method == "net_naive":
        pass
    elif method == "cnn":
        pass
    else:
        print("Method not found!")
    print(f"Method: {method}, The accuracy is: {acc}")


if __name__ == "__main__":
    for method in ["knn", "net"]:
        run(method)
