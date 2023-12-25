"""
@author: laziyu
@date: 2023/11/13
@desc: using pytorch to build a CNN classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# optimizer = torch.optim.Adam(.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DbrhdDataset(Dataset):
    """
    User defined dataset for Dbrhd
    """

    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = torch.from_numpy(images).float().reshape(-1, 1, 32, 32)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


class CNN(nn.Module):
    """
    CNN model
    """
    def __init__(self) -> None:
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # 1, 32, 32 -> 16, 32, 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16, 32, 32 -> 16, 16, 16
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 16, 16, 16 -> 32, 16, 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32, 16, 16 -> 32, 8, 8
        )
        # 输出
        self.out = nn.Linear(32 * 8 * 8, 10)  # 32 * 7 * 7 -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图
        output = self.out(x).float()
        # 输出预测结果
        output = F.softmax(output, dim=1)
        return output

    def fit(self, test_loader):
        """
        pred the test data
        :param test_loader: the test data
        :return: the accuracy
        """
        count = 0
        loss = 0.0
        loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x, y = data[0], data[1]
                x = x.to(device)
                y = y.squeeze().long().to(device)
                output = self(x)
                loss += loss_func(output, y).item()
                pred = torch.max(output, 1)[1]
                count += (pred == y).sum().item()
            loss /= len(test_loader.dataset)
            # print("average loss: ", loss)
        return count / len(test_loader.dataset)
