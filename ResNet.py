import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 16 # 初始通道数调整为更小的尺寸
 
        # 输入层（适应MNIST的1通道输入）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1) # 28x28 -> 14x14
 
        # 残差块层
        self.layer1 = self._make_layer(16, 16, stride=1, blocks=2)
        self.layer2 = self._make_layer(16, 32, stride=2, blocks=2)
        self.layer3 = self._make_layer(32, 64, stride=2, blocks=2)
        self.layer4 = self._make_layer(64, 128, stride=2, blocks=2)
 
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    def _make_layer(self, in_channels, out_channels, stride,blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride, bias=False),nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels,stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels,out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x) # [B,1,28,28] -> [B,16,28,28]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B,16,14,14]
 
        x = self.layer1(x) # [B,16,14,14]
        x = self.layer2(x) # [B,32,7,7]
        x = self.layer3(x) # [B,64,4,4]
        x = self.layer4(x) # [B,128,2,2]
 
        x = self.avgpool(x) # [B,128,1,1]
        x = torch.flatten(x, 1) # [B,128]
        x = self.fc(x) # [B,10]
        return x

myNet = ResNet18()
print(myNet)

#load data
transform=torchvision.transforms.ToTensor()#from image to tensor
from sklearn.datasets import fetch_openml
print('Downloading mnist_784...')
mnist=fetch_openml('mnist_784',as_frame=False)
print('Data downloaded successfully.')
train_data, train_labels = mnist.data[:60000], mnist.target[:60000]
test_data, test_labels = mnist.data[60000:], mnist.target[60000:]#测试集，要统一的。
import numpy as np
# Convert numpy arrays to torch tensors with proper shape and normalization.
train_data = torch.from_numpy(train_data.reshape(-1, 28, 28).astype(np.float32)) / 255.0
train_data = train_data.unsqueeze(1)
test_data = torch.from_numpy(test_data.reshape(-1, 28, 28).astype(np.float32)) / 255.0
train_labels = torch.from_numpy(train_labels.astype(np.int64))
test_labels = torch.from_numpy(test_labels.astype(np.int64))

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

import numpy as np
import matplotlib.pyplot as plt
#def imshow(img):
#    img = img / 2 + 0.5
#    npimg = img.numpy()
#    plt.figure(figsize=(10, 10))
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()

#images, _ = next(iter(train_loader))
#imshow(torchvision.utils.make_grid(images[:64]))

device=torch.device("cpu")
net=ResNet18().to(device)
loss_fuc=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#train
EOPCH=8
x_graph=[]
y_graph=[]
count=0
precision=[]
for epoch in range(EOPCH):
    sum_loss=0
    sum_loss_2=0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        count+=1
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fuc(outputs, labels)
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.item()
        sum_loss_2+=loss.item()
        if i % 10 == 9:
            x_graph.append(count)
            y_graph.append(sum_loss / 10)
            sum_loss = 0.0
        if i%100 ==99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss_2 / 100))
            sum_loss_2=0.0
    correct = 0
    total = 0
    for data in test_loader:
        test_inputs, labels = data
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        test_inputs = test_inputs.view(-1, 1, 28, 28)  # Ensure the input tensor has the correct shape
        outputs_test = net(test_inputs)
        _, predicted = torch.max(outputs_test.data, 1)  #omit the position of max value
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()  
    precision.append(100*correct/total)
for i in range(EOPCH):
    print('#',i+1,': precision:',precision[i])
plt.plot(x_graph,y_graph)
plt.xlabel('number of training(batch)')
plt.ylabel('loss(CrossEntropy)')
plt.grid()
plt.show()  
print('Precision on the entire test set during 8 Epochs:', precision) 
epoch_list=[1,2,3,4,5,6,7,8] 
plt.plot(epoch_list,precision,"-o")
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.show()




