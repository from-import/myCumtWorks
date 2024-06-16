#导入相关库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

#参数定义
#dataset = "FashionMNIST" #选择所使用的数据集"MNIST"or"FashionMNIST"
dataset = "MNIST"
Net = "LeNet5" #选择所使用的网络"LeNet5"or "LinearNet"
# Net = "LinearNet" #选择所使用的网络"LeNet5"or "LinearNet"
batch_size = 64 #定义批处理大小
lr=0.01 #设置learning rate学习率
epochs = 5 #指定训练迭代次数
save_path = "./" #模型保存路径
Early_Stopping = 0 #选择是否使用Early Stopping训练模式，训练时根据精度的变化率来控制训练迭代代数

classes = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 自动选择cpu或gpu用于训练.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

'''
    残差块定义：
    参数：
        in_channels：输入特征的通道数
        out_channels: 输出特征的通道数
        stride: 步长
        downsample: 如果输入输出不匹配，用于下采样的模块
'''


class ResidualBlock(nn.Module):
    # 残差块
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # 进行批量归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 短路连接（如果输入输出通道数或特征图大小不同，需要进行下采样）
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # 主路径
        #         print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 短路连接
        if self.downsample is not None:
            residual = self.downsample(x)

            # 残差连接
        # 将主路径和短路连接的结果相加
        out += residual
        out = self.relu(out)

        # 返回经过残差处理后的特征图
        return out


# 定义模型

if Net == "LinearNet":
    # 普通神经网络
    class LinearNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    model = LinearNet().to(device)

else:
    class CNN_1(nn.Module):
        def __init__(self):
            super(CNN_1, self).__init__()
            self.conv1 = nn.Conv2d(
                ## 卷积核大小为羽3，输入通道为__，输出通道为13，步长为1;
                ## paddingHzero padding
                ##要求经过conv1的输出空间维度与输入的空间维度相同
                in_channels=1,
                out_channels=13,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            ##激活函数+最大池化
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2d = nn.Sequential(
                ## 1. 卷积核大小为3*3，输入通道为10，输出通道为10，padding 方法为same，padding大小为？？？步长为??
                ## 2.自行选择激活函数
                ## 3. 池化
                nn.Conv2d(in_channels=13,
                          out_channels=10,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                self.relu,
                self.pool
            )
            ## 添加残差连接模块
            #############残差模块设计部分################
            self.res_block = ResidualBlock(10, 10)

            ############################################
            self.conv3 = nn.Sequential(
                ##自行设计卷积模块
                nn.Conv2d(in_channels=10,
                          out_channels=29,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                self.relu,
                self.pool
            )
            self.out1 = nn.Linear(3 * 3 * 29, 10, bias=True)
            ## 在下方添加Dropout以及其他代码
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            ## 请将余下代码补充完整
            # 卷积后图像大小保持不变
            # (64,1,28,28)
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            # (64,13,14,14)
            x = self.conv2d(x)
            # (64,10,7,7)
            x = self.res_block(x)
            # (64,10,7,7)
            x = self.conv3(x)
            # (64,10,3,3)

            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.out1(x)

            return x


    model = CNN_1().to(device)

print(model)

# 为了训练模型，我们需要一个损失函数和一个优化器
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器


# 在单个训练循环中，模型对训练数据集进行预测（分批提供给它），并反向传播预测误差以调整模型的参数
def train(dataloader, model, loss_fn, optimizer, flag=0):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        tensor_new = y
        if flag == 1:
            tensor_new = torch.zeros(y.shape[0], 10)
            for i in range(len(y)):
                tensor_new[i, y[i]] = 1
        # Compute prediction error
        tensor_new = tensor_new.to(device)
        pred = model(X)
        #         print(pred.shape)
        #         print(tensor_new.shape)
        loss = loss_fn(pred, tensor_new)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 对照测试数据集检查模型的性能，以确保它正在学习
def test(dataloader, model, loss_fn, flag=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tensor_new = y
            if flag == 1:
                tensor_new = torch.zeros(y.shape[0], 10)
                for i in range(len(y)):
                    tensor_new[i, y[i]] = 1
            tensor_new = tensor_new.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, tensor_new).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# 实现Early Stopping训练
class EarlyStopping:
    def __init__(self, save_path, patience=2, verbose=False, delta=0.03):
        """
        Args:
            save_path : 模型保存路径
            patience (int): 设置将连续几次训练迭代纳入Early Stopping考评
            verbose (bool): 如果是 "True"，则为每次验证损失的优化值打印一条信息
            delta (float): 前后两次训练迭代的最小变化阈值，小于该阈值则认为模型优化幅度有限，将该次迭代计入patience，
                           数量达到patience则提前停止训练。
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        # 尚未找到最佳的验证分数
        if self.best_score is None:
            # 将当前的分数作为最佳分数
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 如果分数没有得到明显改善
        elif score < self.best_score + self.delta:
            # 计时器加1操作
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 如果超出一定时间，说明达到最优，则进行保存
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果得到明显改善，说明还未达到最优，则进行存储，同时继续进行查找
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''当验证损失降低时，保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

from torchvision.transforms import ToTensor, Resize
from PIL import *
# 加载预训练模型
model = LinearNet().to(device) if Net == "LinearNet" else CNN_1().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # 加载模型权重
model.eval()


# 加载预训练模型
model = LinearNet().to(device) if Net == "LinearNet" else CNN_1().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # 加载模型权重
model.eval()

# 加载并预处理图像
image_path = "image1.png"
image = Image.open(image_path).convert('L')  # 打开图像并转为灰度图
resize = Resize((28, 28))  # 调整图像大小为模型输入的大小
transform = ToTensor()
tensor_image = transform(resize(image)).unsqueeze(0).to(device)  # 转换为张量并添加批次维度，移动到对应设备

# 对图像进行预测
with torch.no_grad():
    outputs = model(tensor_image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

# 输出预测结果
print(f'预测结果: "{predicted_class}"')

# 显示图像
plt.imshow(image, cmap='gray')
plt.title(f'Result: "{predicted_class}"')
plt.axis('off')
plt.show()