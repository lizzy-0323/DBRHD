# coding: utf-8
"""
@author: laziyu
@date: 2023/11/13
@desc: using multiple method to recognize the handwritten digits
"""
import os
import time


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import train_test_split as TLS
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.knn import KNN
from model.cnn import CNN, DbrhdDataset
from model.net import NET

# config for cnn
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add this to avoid bug in Mac os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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
        print(f"reading {'train' if train else 'test'} data...")
        for i in tqdm(range(file_num)):
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


def plot(filename, model="knn"):
    """
    plot the result in csv file
    :params filename: the name of the file
    """
    df = pd.read_csv(filename)
    x = df["param"]
    acc = df["acc"]
    plt.plot(x, acc, label="acc")
    plt.legend()
    # plt.show()
    # save fig
    if not os.path.exists("./result"):
        os.mkdir("./result")
    plt.savefig(f"./result/{model}.png")


def benchmark(methods=None):
    """
    run mutiple algorithm and compare the result and time
    """
    if methods is None:
        methods = ["knn", "net", "net_naive", "cnn"]
    for method in methods:
        start_time = time.time()
        if method == "knn":
            acc, wrong_num = run_knn(k=3)
        elif method == "net":
            acc, wrong_num = run_net(lr=0.01, hidden_num=500)
        elif method == "net_naive":
            acc, wrong_num = run_net_naive(hidden_num=500)
        elif method == "cnn":
            acc, wrong_num = run_cnn(lr=0.01)
        else:
            raise ValueError("method not found!")
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        with open("./result/benchmark.csv", "a", encoding="utf-8") as f:
            f.write(f"{method},{acc},{wrong_num},{end_time - start_time}\n")


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


def run_net(lr=0.01, hidden_num=500):
    """
    run neural network
    :params lr: learning rate
    :params hidden_num: the number of hidden layer
    :return acc: accuracy
    :return wrong_num: the number of wrong prediction
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    train_data, valid_data, train_labels, valid_labels = TLS(
        train_data, train_labels, test_size=0.2, random_state=420
    )
    dnn = DNN(
        hidden_layer_sizes=(hidden_num,), random_state=420, learning_rate_init=lr
    ).fit(train_data, train_labels.ravel())
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
    dbrhd_train = DbrhdDataset(train_data, train_labels)
    dbrhd_test = DbrhdDataset(test_data, test_labels)
    train_loader = DataLoader(dbrhd_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dbrhd_test, batch_size=BATCH_SIZE, shuffle=True)
    # init cnn model
    cnn = CNN()
    # init optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    # init loss function
    loss_func = nn.CrossEntropyLoss()
    epochs = EPOCH
    for epoch in range(epochs):
        cnn.train()
        print(f"lr: {lr} ")
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


def run_net_naive(hidden_num=500):
    """
    run numpy neural network
    :return acc: accuracy
    :return wrong_num: the number of wrong prediction
    """
    train_data, train_labels = load_data("./data", train=True)
    test_data, test_labels = load_data("./data", train=False)
    input_size = train_data.shape[1]
    net = NET(input_size=input_size, hidden_size=hidden_num)
    np.random.seed(420)
    np.random.shuffle(train_data)
    np.random.shuffle(train_labels)
    for epoch in range(EPOCH):
        # 按照batch_size进行训练
        step = 0
        for i in range(0, input_size, BATCH_SIZE):
            x = train_data[i : i + BATCH_SIZE]
            y = train_labels[i : i + BATCH_SIZE]
            output = net(x)
            loss = net.loss(output, y)
            net.backward(x, y)
            net.update_grad()
            net.update_learning_rate()
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
            step += 1
    acc = net.fit(test_data, test_labels)
    wrong_num = len(test_labels) - int(acc * len(test_labels))
    return acc, wrong_num


def run(method=None, save=True, params=None, param_type="k"):
    """
    run the program
    :param methods: the method to recognize the handwritten digits
    :param save: whether to save the result
    :param params: the parameters of the methods
    :param param_type: the type of the parameters
    """
    acc = 0
    wrong_num_list = []
    acc_list = []
    if params is None:
        raise ValueError("Please input the params!")
    if method == "knn":
        for k in params:
            acc, wrong_num = run_knn(k)
            acc_list.append(acc)
            wrong_num_list.append(wrong_num)
    elif method == "net":
        for param in params:
            if param_type == "lr":
                acc, wrong_num = run_net(lr=param, hidden_num=500)
            elif param_type == "hidden_num":
                acc, wrong_num = run_net(lr=0.01, hidden_num=param)
            else:
                raise ValueError("param_type not found!")
            acc_list.append(acc)
            wrong_num_list.append(wrong_num)
    elif method == "net_naive":
        for param in params:
            acc, wrong_num = run_net_naive(hidden_num=param)
            acc_list.append(acc)
            wrong_num_list.append(wrong_num)
    elif method == "cnn":
        for param in params:
            acc, wrong_num = run_cnn(lr=param)
            acc_list.append(acc)
            wrong_num_list.append(wrong_num)
    else:
        raise ValueError("method not found!")
    print("algorithm done")

    # if plot the result
    if save:
        save_result(method, params, acc_list, wrong_num_list)
    # for test not for benchmark
    # print(f"Method: {method}, The accuracy is: {acc}")


if __name__ == "__main__":
    # run("net", save=True, params=[0.1, 0.01, 0.001, 0.0001], param_type="lr")
    # run("net_naive", save=True, params=[500, 1000, 1500, 2000], param_type="hidden_num")
    # plot("./result/cnn.csv", model="cnn")
    benchmark(["knn", "net", "net_naive", "cnn"])
