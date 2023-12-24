"""
@author: laziyu
@date: 2023/11/14
@desc: using numpy to build a neural network
"""
import numpy as np

# config for net
LR = 0.001
LEARNING_RATE_DECAY = 0.9


def softmax(x):
    """softmax function"""
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def change_one_hot(labels):
    """change label to one-hot encoding mode"""
    label_one_hot = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        label_one_hot[i, int(label)] = 1.0
    return label_one_hot


def cross_entropy(y, y_hat):
    """cross entropy loss"""
    return -np.sum(y * np.log(y_hat + 1e-7)) / len(y)


class NET:
    """numpy naive neural network, two fc layers"""

    def __init__(
        self,
        input_size=1024,
        hidden_size=500,
        output_size=10,
        weight_init_std=0.01,
    ) -> None:
        """
        :param input_size: the size of input data
        :param hidden_size: the size of hidden layer
        :param output_size: the size of output data
        :param weight_init_std: the standard deviation of weight initialization, for better convergence
        """
        self.learning_rate = LR
        self.learning_rate_decay = LEARNING_RATE_DECAY
        self.params = {}
        self.grads = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def __call__(self, param):
        return self.forward(param)

    def forward(self, x):
        """
        forward propagation
        :param x: input data
        """
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]
        fc1 = np.dot(x, w1) + b1  # 64, 1024 -> 64, 500
        h1 = sigmoid(fc1)  # 64, 500
        fc2 = np.dot(h1, w2) + b2  # 64, 500 -> 64, 10
        return softmax(fc2)  # 64, 10

    def backward(self, x, label):
        """
        backward propagation
        :param x: input data
        :param label: input label
        """
        # forward
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
        """update the gradient"""
        self.params["W1"] -= self.learning_rate * self.grads["W1"]
        self.params["b1"] -= self.learning_rate * self.grads["b1"]
        self.params["W2"] -= self.learning_rate * self.grads["W2"]
        self.params["b2"] -= self.learning_rate * self.grads["b2"]

    def fit(self, x, labels):
        """
        test the accuracy of the model
        :param x: test data
        :param labels: test labels
        """
        y = self.forward(x)
        y = np.squeeze(y).astype(int)
        # 由于求index，所以可以不用one-hot
        pred = np.argmax(y, axis=1)
        acc = np.sum(pred == labels) / len(labels)
        return acc

    def loss(self, output, labels):
        """
        calculate the loss
        :param output: the output of the model
        :param labels: the labels of the data
        :return: the loss
        """
        # 将label转换为one-hot
        one_hot_labels = change_one_hot(labels)
        # 计算交叉熵损失
        loss = cross_entropy(one_hot_labels, output)
        return loss

    def update_learning_rate(self):
        """learning rate decay"""
        self.learning_rate *= self.learning_rate_decay
