# coding: utf-8
"""
@author: laziyu
@date: 2023/11/13
@desc: using multiple method to recognize the handwritten digits
"""
import os
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import train_test_split as TLS


from model.knn import KNN
from model.cnn import CNN, DbrhdDataset
from model.net import NET

# config for cnn
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add this to avoid bug in Mac os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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


def plot(filename):
    """
    plot the result in csv file
    :params filename: the name of the file
    """
    ...


def save_result(filename, param_list, acc_list, wrong_num_list):
    """
    save result to csv file
    :params param_list: x-axis
    :params acc_list: acc list
    :params wrong_num_list: wrong num list
    """
    if not os.path.exists("./result"):
        os.mkdir("./result")
    with open(f"./result/{filename}.csv", "w", encoding="utf-8") as f:
        f.write("param,acc,wrong_num\n")
        for i, param in enumerate(param_list):
            f.write(f"{param},{acc_list[i]},{wrong_num_list[i]}\n")


def run_knn(k):
    """
    run knn algorithm
    :params k: the number of nearest neighbors
    :return acc: accuracy
    :return wrong_num: the number of wrong prediction
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    knn = KNN(train_data, train_labels, k)
    acc, wrong_num = knn.fit(test_data, test_labels, k)
    return acc, wrong_num


def run_net(lr):
    """
    run neural network
    :params lr: learning rate
    :return acc: accuracy
    :return wrong_num: the number of wrong prediction
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    train_data, valid_data, train_labels, valid_labels = TLS(
        train_data, train_labels, test_size=0.2, random_state=420
    )
    dnn = DNN(hidden_layer_sizes=(500,), random_state=420, learning_rate_init=lr).fit(
        train_data, train_labels.ravel()
    )
    # 采用交叉验证
    # print(dnn.score(valid_data, valid_labels))
    acc = dnn.score(test_data, test_labels)
    wrong_num = len(test_labels) - int(acc * len(test_labels))
    return acc, wrong_num


def run_cnn(lr):
    """
    run cnn network
    :params lr: learning rate
    :return acc: accuracy
    :return wrong_num: the number of wrong prediction
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    train_data = DbrhdDataset(train_data, train_labels)
    test_data = DbrhdDataset(test_data, test_labels)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # init cnn model
    cnn = CNN()
    # init optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    # init loss function
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
    wrong_num = len(test_labels) - int(acc * len(test_labels))
    return acc, wrong_num


def run_net_naive():
    ...


def run(methods=None, save=True):
    """
    run the program
    :param methods: the methods to recognize the handwritten digits
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    acc = 0
    wrong_num_list = []
    acc_list = []
    if methods is None:
        methods = ["knn", "net", "cnn", "net_naive"]
    for method in methods:
        if method == "knn":
            params = [1, 3, 5, 7]
            for k in params:
                acc, wrong_num = run_knn(k)
                acc_list.append(acc)
                wrong_num_list.append(wrong_num)
        elif method == "net":
            params = [0.1, 0.01, 0.001, 0.0001]
            for lr in params:
                acc, wrong_num = run_net(lr)
                acc_list.append(acc)
                wrong_num_list.append(wrong_num)
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
            # params = [0.1, 0.01, 0.001, 0.0001]
            # for lr in params:
            #     acc, wrong_num = run_cnn(lr)
            #     acc_list.append(acc)
            #     wrong_num_list.append(wrong_num)
            # train_data, train_labels = load_data("./data", train=True)
            # test_data, test_labels = load_data("./data", train=False)
            train_data = DbrhdDataset(train_data, train_labels)
            test_data = DbrhdDataset(test_data, test_labels)
            print(233)
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
            # init cnn model
            cnn = CNN()
            # init optimizer
            optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
            # init loss function
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
        # if plot the result
        if save:
            save_result(method, params, acc_list, wrong_num_list)
        print("algorithm done")
        # for test not for benchmark
        # print(f"Method: {method}, The accuracy is: {acc}")


if __name__ == "__main__":
    run(["cnn"])
