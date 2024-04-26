import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader


class Net(nn.Module):  # 定义了一个名为 Net 的类，它继承自 nn.Module
    def __init__(self):  # 构造函数，3x32x32的卷积层，2x2的池化层，64个全连接层
        super(Net, self).__init__()  # 调用父类的构造函数

        # 定义网络结构
        self.conv1 = nn.Conv2d(3,6,3)  # 第一个卷积层，输入通道为3，输出通道为6，卷积核大小为3x3
        self.conv2 = nn.Conv2d(6,16,3)  # 第二个卷积层，输入通道为6，输出通道为16，卷积核大小为3x3
        self.fc1 = nn.Linear(16*28*28, 512)  # 第一个全连接层，输入特征数为16*28*28，输出特征数为512
        self.fc2 = nn.Linear(512,64)  # 第二个全连接层，输入特征数为512，输出特征数为64
        self.fc3 = nn.Linear(64,10)  # 第三个全连接层，输入特征数为64，输出特征数为10

    def forward(self, x):
        # 定义前向传播过程
        x=self.conv1(x)  # 卷积层 1
        x=F.relu(x)  # 使用 ReLU 激活函数

        x=self.conv2(x)  # 卷积层 2
        x=F.relu(x)  # 使用 ReLU 激活函数

        x=x.view(-1,16*28*28)  # 对输入进行展平操作
        x=self.fc1(x)  # 全连接层 1
        x=F.relu(x)  # 使用 ReLU 激活函数

        x=self.fc2(x)  # 全连接层 2
        x=F.relu(x)  # 使用 ReLU 激活函数

        x=self.fc3(x)  # 全连接层 3
        return x  # 返回输出


net = Net()  # 创建一个名为net的神经网络实例
print(net)  # 打印神经网络实例



import torch.optim as optim
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # 定义优化器为随机梯度下降，学习率为0.1

# mini batch训练神经网络
for epoch in range(2):
    for i, data in enumerate(trainloader):
        images, labels = data
        outputs = net(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if (i % 1000 == 0):
            print('Epoch: %d, Step: /%d, Loss: %.3f' % (epoch, i, loss.item()))