import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
import torchvision
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #conv1,pool1,6*28*28->6*14*14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  #kernel 2*2,move 2 pixels per time
        )
        #conv2,pool2 6*14*14->16*10*10->16*5*5
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        #full connection1
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        #full connection2
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        #full connection3
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)  #flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

myNet = LeNet5()
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
net=LeNet5().to(device)
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




