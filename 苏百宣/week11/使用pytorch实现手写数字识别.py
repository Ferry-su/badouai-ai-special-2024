from os import supports_fd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tensorflow import truediv

from tf import model


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimist = self.create_optimizer(optimist)

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),  # 交叉熵损失
            'MSE': nn.MSELoss()  # 均方误差损失
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optimizer = {
        'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
        'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
        'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests),
        }
        return support_optimizer[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            runing_loss = 0.0
            for i , data in enumerate(train_loader,0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimozer.step()

                runing_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i+1)*1./len(train_loader), runing_loss / 100))
                    runing_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating on test set')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

def mnist_load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像或 numpy 数组转换为 Tensor
        transforms.Normalize((0.5,), (0.5,))  # 使用均值和标准差对单通道数据进行标准化
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=truediv(), num_workers=2)
    return trainloader, testloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTRORY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
