import numpy as np

class NET:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # 初始化权重和偏置
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1)
        self.bias_input_hidden1 = np.zeros((1, hidden_size1))
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias_hidden1_hidden2 = np.zeros((1, hidden_size2))
        self.weights_hidden2_output = np.random.randn(hidden_size2, output_size)
        self.bias_hidden2_output = np.zeros((1, output_size))

        # 用于存储中间结果的变量
        self.input_layer = None
        self.hidden_layer1 = None
        self.hidden_layer2 = None
        self.output_layer = None
        self.predictions = None

    def forward(self, input_data):
        # 前向传播
        self.input_layer = input_data
        self.hidden_layer1 = np.dot(self.input_layer, self.weights_input_hidden1) + self.bias_input_hidden1
        self.hidden_layer1_activation = self.relu(self.hidden_layer1)
        self.hidden_layer2 = np.dot(self.hidden_layer1_activation, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2
        self.hidden_layer2_activation = self.relu(self.hidden_layer2)
        self.output_layer = np.dot(self.hidden_layer2_activation, self.weights_hidden2_output) + self.bias_hidden2_output
        self.predictions = self.softmax(self.output_layer)
        return self.predictions

    def backward(self, target, learning_rate=0.01):
        # 反向传播
        output_error = target - self.predictions
        output_delta = output_error

        hidden2_error = output_delta.dot(self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * self.relu_derivative(self.hidden_layer2_activation)

        hidden1_error = hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * self.relu_derivative(self.hidden_layer1_activation)

        # 更新权重和偏置
        self.weights_hidden2_output += self.hidden_layer2_activation.T.dot(output_delta) * learning_rate
        self.bias_hidden2_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_hidden1_hidden2 += self.hidden_layer1_activation.T.dot(hidden2_delta) * learning_rate
        self.bias_hidden1_hidden2 += np.sum(hidden2_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden1 += self.input_layer.T.dot(hidden1_delta) * learning_rate
        self.bias_input_hidden1 += np.sum(hidden1_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(y, learning_rate)
            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(y, predictions)
                print(f'Epoch {epoch}, Loss: {loss}')

    def _call__(self, input_data):
        # 调用 __call__ 方法时自动调用 forward 方法
        return self.forward(input_data)

    def calculate_accuracy(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# 使用示例
if __name__ == "__main__":
    # 假设输入大小为1024，两个隐藏层的神经元数量分别为256和128，输出大小为10（0-9）
    net = NET(1024, 256, 128, 10)

    # 假设你有训练数据X_train和标签y_train
    X_train = np.random.rand(1934, 1024)
    y_train = np.random.randint(0, 10, (1934, 1))
    y_train_onehot = np.eye(10)[y_train.flatten()]

    # 进行训练
    net.train(X_train, y_train_onehot, epochs=1000)

    # 测试数据
    X_test = np.random.rand(976, 1024)
    y_test = np.random.randint(0, 10, (976, 1))
    y_test_onehot = np.eye(10)[y_test.flatten()]

    # 计算准确率
    accuracy = net.calculate_accuracy(X_test, y_test_onehot)
    print(f'Accuracy on test data: {accuracy}')
