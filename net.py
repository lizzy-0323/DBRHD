"""
@author: laziyu
@date: 2023/11/13
@desc: using numpy to build a neural network
"""
import numpy as np

LR = 0.001
LEARNING_RATE_DECAY = 0.9


def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def change_one_hot(labels):
    label_one_hot = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        label_one_hot[i, int(label)] = 1.0
    return label_one_hot


class NET(object):
    def __init__(
        self,
        input_size=1024,
        hidden_size=500,
        output_size=10,
        weight_init_std=0.01,
    ) -> None:
        self.learning_rate = LR
        self.learning_rate_decay = LEARNING_RATE_DECAY
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def __call__(self, param):
        return self.forward(param)

    def forward(self, x):
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]
        # 第一个全连接层
        fc1 = np.dot(x, w1) + b1
        h1 = sigmoid(fc1)
        # 第二个全连接层
        fc2 = np.dot(h1, w2) + b2
        return softmax(fc2)

    def backward(self, x, label):
        # forward
        self.grads = {}
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]
        a1 = np.dot(x, w1) + b1
        h1 = sigmoid(a1)
        a2 = np.dot(h1, w2) + b2
        output = softmax(a2)
        # backward
        dy = (output - label) / x.shape[0]
        self.grads["W2"] = np.dot(h1.T, dy)
        self.grads["b2"] = np.sum(dy, axis=0)
        da1 = np.dot(dy, w2.T)
        ha1 = sigmoid(a1)
        dz1 = (1.0 - ha1) * ha1 * da1
        self.grads["W1"] = np.dot(x.T, dz1)
        self.grads["b1"] = np.sum(dz1, axis=0)

    def update_grad(self):
        self.params["W1"] -= self.learning_rate * self.grads["W1"]
        self.params["b1"] -= self.learning_rate * self.grads["b1"]
        self.params["W2"] -= self.learning_rate * self.grads["W2"]
        self.params["b2"] -= self.learning_rate * self.grads["b2"]

    def fit(self, x, labels):
        y = self.forward(x)
        y = np.squeeze(y).astype(int)
        y = np.argmax(y, axis=1)
        acc = np.sum(y == labels) / len(labels)
        return acc

    def loss(self, output, labels):
        # 将label转换为one-hot
        labels_one_hot = change_one_hot(labels)
        # 计算交叉熵损失
        loss = -np.sum(labels_one_hot * np.log(output + 1e-7)) / len(labels)
        return loss

    def update_learning_rate(self):
        self.learning_rate *= self.learning_rate_decay
