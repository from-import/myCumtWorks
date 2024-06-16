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

#下载数据集
if dataset=="MNIST":
    # 从torchvision下载训练集.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # #从torchvision下载测试集.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # 若是数字数据集则设置成下述类别
    classes = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
else:
    # 从torchvision下载训练集.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # #从torchvision下载测试集.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # 若是服饰数据集则设置成下述类别
    classes = [ "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal","Shirt", "Sneaker", "Bag", "Ankle boot"]

# 创建数据加载器.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#观察数据样本
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    fig = plt.figure()
    # 对部分数据进行显示
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(X[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(classes[int(y[i])]))
      plt.xticks([])
      plt.yticks([])
    plt.show()
    break
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
    # 残差块 梯度在反向传播过程中可能会逐渐消失
    # 在深层网络中，随着反向传播的进行，梯度可能会逐层缩小，最终在靠近输入层时几乎变为零。这会导致权重更新变得非常慢，网络难以训练。

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
        # 保存输入数据，用于短路连接
        residual = x

        # 主路径（Main Path）
        # 对输入进行第一次卷积操作
        out = self.conv1(x)
        # 对卷积结果进行批量归一化
        out = self.bn1(out)
        # 对归一化结果进行ReLU激活
        out = self.relu(out)
        # 对激活结果进行第二次卷积操作
        out = self.conv2(out)
        # 对卷积结果进行批量归一化
        out = self.bn2(out)

        # 短路连接（Shortcut Connection）
        # 如果输入和输出的尺寸不匹配，通过下采样调整输入尺寸
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将主路径的输出与短路连接的输出相加，实现残差连接
        out = out + residual
        # 对相加结果进行ReLU激活 对于输入小于0的值输出0，对于大于0的值输出原值 创造非线性
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
            # 第一层卷积层
            self.conv1 = nn.Conv2d(
                in_channels=1,  # 输入通道数为1（灰度图像）
                out_channels=13,  # 输出通道数为13
                kernel_size=3,  # 卷积核大小为3x3
                stride=1,  # 步长为1
                padding=1,  # 使用填充，保持输入输出尺寸相同
            )
            # 激活函数和最大池化层
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 缩小,减少冗余

            # 第二层卷积层
            self.conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=13,  # 输入通道数为13
                    out_channels=10,  # 输出通道数为10
                    kernel_size=3,  # 卷积核大小为3x3
                    stride=1,  # 步长为1
                    padding=1  # 使用填充，保持输入输出尺寸相同
                ),
                self.relu,
                self.pool
            )

            # 添加残差连接模块
            self.res_block = ResidualBlock(10, 10)

            # 第三层卷积层
            self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=10,  # 输入通道数为10
                    out_channels=29,  # 输出通道数为29
                    kernel_size=3,  # 卷积核大小为3x3
                    stride=1,  # 步长为1
                    padding=1  # 使用填充，保持输入输出尺寸相同
                ),
                self.relu,
                self.pool
            )

            # 全连接层 0123456789
            self.out1 = nn.Linear(3 * 3 * 29, 10, bias=True)

            # Dropout层 随机去掉神经元 防止梯度消失
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.relu(self.conv1(x))  # 卷积层1后接激活函数ReLU
            x = self.pool(x)  # 最大池化层
            x = self.conv2d(x)  # 卷积层2
            x = self.res_block(x)  # 残差连接模块
            x = self.conv3(x)  # 卷积层3

            x = x.view(x.size(0), -1)  # 将特征图展平
            x = self.dropout(x)  # Dropout层
            x = self.out1(x)  # 全连接层

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



for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test_loss = test(test_dataloader, model, loss_fn)

print("Done!")



#保存模型的一种常见方法是序列化内部状态字典(包含模型参数)
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


#加载模型的过程包括重新创建模型结构并将状态字典加载到其中。
model = LinearNet().to(device) if Net == "LinearNet" else CNN_1().to(device)
model.load_state_dict(torch.load("model.pth"))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze_(0).to(device)
#     x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# 显示卷积核
# 卷积核可视化
# 设置存储路径
baseDir = os.path.dirname("./")
log_dir = os.path.join(baseDir, "result")
writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")

# 设置权重文件的路径
state_dict_path = "./model.pth"
# 判断路径是否存在
if os.path.exists(state_dict_path):
    # 如果权重文件存在，则把他加载进来
    print("model path exists")
    visual_model = CNN_1().to(device)
    state_dict = torch.load(state_dict_path)
    visual_model.load_state_dict(state_dict)
else:
    print("model path dosen exist")

# 想要访问的起始网络层的序号
kernel_num = -1

# 最多可视化到的网络层序号
vis_max = 1

# 遍历网络层
for sub_module in visual_model.modules():
    # 判断是否为卷积层
    if isinstance(sub_module, nn.Conv2d):
        kernel_num += 1
        # 判断是否超过要可视化的网络层序号
        if kernel_num > vis_max:
            break
        kernels = sub_module.weight
        # 得到输出通道数，输入通道数，核宽，核高
        c_out, c_int, k_w, k_h = tuple(kernels.shape)
        print(kernels.shape)
        # 对c_out单独可视化
        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # make_grid需要 BCHW，这里拓展C维度

            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid)

        # 对64个卷积核直接进行可视化
        kernel_all = kernels.view(-1, 1, k_h, k_w)  # b，3, h, w
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=322)
        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))
writer.close()
