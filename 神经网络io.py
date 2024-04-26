import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):  # 定义了一个名为 Net 的类，它继承自 nn.Module
    def __init__(self):  # 构造函数
        super(Net, self).__init__()  # 调用父类的构造函数

        # 定义网络结构
        self.conv1 = nn.Conv2d(1,6,3)  # 第一个卷积层，输入通道为1，输出通道为6，卷积核大小为3x3
        self.conv2 = nn.Conv2d(6,16,3)  # 第二个卷积层，输入通道为6，输出通道为16，卷积核大小为3x3
        self.fc1 = nn.Linear(16*28*28,512)  # 第一个全连接层，输入特征数为16*28*28，输出特征数为512
        self.fc2 = nn.Linear(512,64)  # 第二个全连接层，输入特征数为512，输出特征数为64
        self.fc3 = nn.Linear(64,2)  # 第三个全连接层，输入特征数为64，输出特征数为2

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


# 创建一个输入数据张量
input_data = torch.randn(1, 1, 32, 32)
print(input_data)
print(input_data.size())

# 使用神经网络处理输入数据
out = net(input_data)
print(out)
print(out.size())

# 创建目标张量，并对其进行变形
target = torch.randn(2)
target = target.view(1, -1)
print(target)

criterion = nn.L1Loss()  # 使用 L1 损失函数
loss = criterion(out, target)  # 计算损失值
print(loss)

net.zero_grad()  # 清空梯度
loss.backward(retain_graph=True)  # 反向传播，并保留计算图

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 使用 SGD 优化器,学习速率为0.01
optimizer.step()  # 更新参数

out=net(input_data)  # 再次使用神经网络处理输入数据
print(out)
print(out.size())
