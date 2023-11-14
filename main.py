# coding: utf-8
"""
@author: laziyu
@date: 2023/11/13
@desc: using multiple method to recognize the handwritten digits
"""
import os
import torch
import torch.nn as nn
import numpy as np
from knn import KNN
from cnn import CNN, DbrhdDataset
from net import NET
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import train_test_split as TLS
from torch.utils.data import DataLoader

# config for cnn
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run(methods=None):
    """
    run the program
    :param methods: the methods to recognize the handwritten digits
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    acc = 0
    if methods is None:
        methods = ["knn", "net", "cnn"]
    for method in methods:
        if method == "knn":
            knn = KNN(train_data, train_labels, k=3)
            acc = knn.fit(test_data, test_labels, k=3)
        elif method == "net":
            train_data, valid_data, train_labels, valid_labels = TLS(
                train_data, train_labels, test_size=0.2, random_state=420
            )
            dnn = DNN(hidden_layer_sizes=(500,), random_state=420).fit(
                train_data, train_labels.ravel()
            )
            # 采用交叉验证
            # print(dnn.score(valid_data, valid_labels))
            acc = dnn.score(test_data, test_labels)
        elif method == "net_naive":
            epochs = EPOCH
            batch_size = BATCH_SIZE
            input_size = train_data.shape[1]
            net = NET(input_size=input_size, hidden_size=500)
            np.random.seed(420)
            np.random.shuffle(train_data)
            np.random.shuffle(train_labels)
            for epoch in range(epochs):
                # 按照batch_size进行训练
                step = 0
                for i in range(0, input_size, batch_size):
                    x = train_data[i : i + batch_size]
                    y = train_labels[i : i + batch_size]
                    output = net(x)
                    loss = net.loss(output, y)
                    net.backward(x, y)
                    net.update_grad()
                    net.update_learning_rate()
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
                    step += 1
            acc = net.fit(test_data, test_labels)
        elif method == "cnn":
            cnn = CNN()
            train_data = DbrhdDataset(train_data, train_labels)
            test_data = DbrhdDataset(test_data, test_labels)
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
            # 初始化优化器
            optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
            # 初始化损失函数
            loss_func = nn.CrossEntropyLoss()
            epochs = EPOCH
            for epoch in range(epochs):
                cnn.train()
                for step, data in enumerate(train_loader):
                    x, y = data[0], data[1]
                    x = x.to(device)
                    y = y.squeeze().long().to(device)
                    optimizer.zero_grad()
                    outputs = cnn(x)
                    loss = loss_func(outputs, y)
                    loss.backward()
                    optimizer.step()
                    # 打印输出
                    if step % 10 == 0:
                        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
            cnn.eval()
            acc = cnn.fit(test_loader)

        else:
            print("Method not found!")
        print(f"Method: {method}, The accuracy is: {acc}")


if __name__ == "__main__":
    run(["net_naive"])
