import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # 数据加载和转换
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    batch_size = 128

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    #print(trainloader)

    # 加载验证数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    vgg19_bn = torchvision.models.vgg19_bn(num_classes=10)

    # 使用torchsummary来显示模型结构
    #summary(vgg19_bn, (3, 32, 32))  # CIFAR-10图像大小为32x32x3
    
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(vgg19_bn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 随机梯度下降优化器

    # 训练模型并记录训练/验证准确度和损失
    num_epochs = 40
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    best_accuracy = 0.0  # 用于保存具有最高验证准确度的模型权重

    for epoch in range(num_epochs):
        vgg19_bn.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #print(i)
            optimizer.zero_grad()  # 清零梯度
            outputs = vgg19_bn(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 统计正确预测数量
        
        
        train_loss.append(running_loss / (i + 1))
        train_acc.append(100 * correct / total)
        
        # 验证模型
        vgg19_bn.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                #inputs, labels = inputs.cuda(), labels.cuda()  # 将数据移到GPU上
                
                outputs = vgg19_bn(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        valid_loss.append(running_loss / (i + 1))
        accuracy = 100 * correct / total
        valid_acc.append(accuracy)
        
        # 保存具有最高验证准确度的模型权重
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(vgg19_bn.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}%, Valid Loss: {valid_loss[-1]:.4f}, Valid Acc: {valid_acc[-1]:.2f}%")

    print('训练完成')

    # 绘制并保存训练和验证准确度和损失的图表
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(valid_acc, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('training_plot.png')
