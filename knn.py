"""
@author: laziyu
@date: 2023/11/12
@desc: using numpy to build a knn classifier    
"""
import numpy as np


class KNN(object):
    def __init__(self, train_data, train_label, k) -> None:
        """
        knn algorithm
        :param train_data: the training data
        :param train_label: the training label
        """
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = None
        self.valid_label = None
        self.k = k
        self.dataset_partition()
        # self.build()

    def dataset_partition(self, rate=0.8):
        """
        :param rate: the rate of training set
        """
        # 打乱数据集
        index = np.arange(len(self.train_data))
        np.random.shuffle(index)
        self.train_data = self.train_data[index]
        self.train_label = self.train_label[index]
        # 划分数据集
        self.valid_data = self.train_data[int(rate * len(self.train_data)) :]
        self.valid_label = self.train_label[int(rate * len(self.train_label)) :]
        self.train_data = self.train_data[: int(rate * len(self.train_data))]
        self.train_label = self.train_label[: int(rate * len(self.train_label))]

    def build(self):
        """
        build the knn model, but actually it is not necessary
        """
        count = 0
        for data, label in zip(self.valid_data, self.valid_label):
            if self.classify(data, self.k) == label:
                count += 1
        print("Building knn finished!")
        # print("The accuracy on valid set is: ", count / len(self.valid_label))

    def classify(self, test_data, k):
        """
        classify the test data
        :param test_data: the test data
        :param k: the number of nearest neighbors
        :return: the label of the test data
        """
        distance_list = []
        for data, label in zip(self.train_data, self.train_label):
            distance_list.append((self.distance(data, test_data), label))
        distance_list.sort(key=lambda x: x[0])
        label_list = [i[1] for i in distance_list[:k]]
        return max(label_list, key=label_list.count)

    def distance(self, i, j):
        """
        return the European distance between i and j
        :param i: the first vector
        :param j: the second vector
        :return: the distance between i and j
        """
        return np.sqrt(np.sum(np.square(i - j)))

    def fit(self, test_data, test_label, k):
        """
        fit the model
        :param test_data: the test data
        :param test_label: the test label
        :param k: the number of nearest neighbors
        """
        # 将验证集和训练集合并
        count = 0
        # self.train_data = np.vstack((self.train_data, self.valid_data))
        # self.train_label = np.vstack((self.train_label, self.valid_label))
        for data, label in zip(test_data, test_label):
            if self.classify(data, k) == label:
                count += 1
        return count / len(test_label)
