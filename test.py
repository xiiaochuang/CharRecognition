import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

# 定义超参数
batch_size = 1  # 批的大小

# 下载训练集 MNIST 手写数字测试集，10000个样本
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载 Train 模型
model = torch.load('cnn.pt')
criterion = nn.CrossEntropyLoss()
model.eval()
eval_acc = 0
eval_loss = 0

# 测试
for data in test_loader:
    img, label = data
    # 可视化图像
    plt.figure()
    plt.imshow(img[0][0])
    plt.show()
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)

    _, pred = torch.max(out, 1)
    print(pred)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
    print('Test Loss: {:.6f}   ,   Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))


