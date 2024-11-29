import torch
import CNN
from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 定义超参数
learning_rate = 1e-2  # 学习率
batch_size = 32  # 批的大小
epoches_num = 20  # 遍历训练集的次数

# 下载训练集 MNIST 手写数字训练集，60000个样本
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义model 、loss 、optimizer
model = CNN.SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    print("CUDA is enable!")
    model = model.cuda()
    model.train()

# 开始训练
for epoch in range(epoches_num):
    print('*' * 40)
    train_loss = 0.0
    train_acc = 0.0

    # 训练
    for i, data in enumerate(train_loader, 1):
        img, label = data

        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # 前向传播
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 损失/准确率计算
        train_loss += loss.item() * label.size(0)
        _, pred = out.max(1)
        num_correct = pred.eq(label).sum()
        accuracy = pred.eq(label).float().mean()
        train_acc += num_correct.item()

    print('Finish  {}  Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, train_loss / len(train_dataset),
                                                         train_acc / len(train_dataset)))

# 保存模型
torch.save(model, 'cnn.pt')

