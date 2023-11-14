"""
@author: laziyu
@date: 2023/11/13
@desc: using numpy to build a neural network
"""
import numpy as np

LR = 0.001
LEARNING_RATE_DECAY = 0.9


class FullyConnectedLayer(object):
    """
    Fully connected layer
    """

    def __init__(self, input_size, hidden_size, weight_init_std=0.01) -> None:
        """
        :param input_size: the size of input
        :param hidden_size: the size of hidden layer
        :param learning_rate: the learning rate
        """
        self.params = {}
        self.params["W"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b"] = np.zeros(hidden_size)
        self.input = None
        self.output = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.params["W"]) + self.params["b"]
        return self.output

    def backward(self, top_diff):
        # 计算损失函数对W的导数
        self.dW = np.dot(self.input.T, top_diff)
        # 计算损失函数对b的导数
        self.db = np.sum(top_diff, axis=0, keepdims=True)
        # 计算损失对于输入的导数
        dx = np.dot(top_diff, self.params["W"].T)
        return dx

    def update(self, learning_rate):
        # print("gradient update")
        self.params["W"] -= learning_rate * self.dW
        self.params["b"] -= learning_rate * self.db


class SigmoidLayer(object):
    """
    sigmoid layer
    """

    def __init__(self) -> None:
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, top_diff):
        # 计算导数
        return top_diff * self.output * (1 - self.output)


def loss_function(y, y_pred):
    """
    cross entropy loss function
    :param y: the label
    :param y_pred: the predict label
    """
    batch_size = y_pred.shape[0]
    # 加入1e-7防止log(0)的出现
    return -np.sum(y * np.log(y_pred + 1e-7)) / batch_size


class SoftmaxLayer(object):
    """
    softmax layer
    """

    def __init__(self) -> None:
        self.loss_func = loss_function
        self.output = None
        self.label_one_hot = None

    def forward(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            self.output = y.T
            return self.output
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def get_loss(self, output, label):
        batch_size = output.shape[0]
        # 创建一个one-hot label数组
        self.label_one_hot = np.zeros_like(output)
        self.label_one_hot[np.arange(batch_size), label] = 1.0
        loss = self.loss_func(output, self.label_one_hot)
        return loss

    def backward(self):
        batch_size = self.label_one_hot.shape[0]
        grad = self.output - self.label_one_hot
        dx = grad / batch_size
        return dx


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
