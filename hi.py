import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import numpy as np
import math

# Нейронная сеть

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(13, 24)
        self.fc2 = nn.Linear(24, 48) # В i слое 120 узлов, в i+1 84 узла
        self.fc3 = nn.Linear(48, 51)



    def forward(self, x):
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net()


#Осуществляем оптимизацию путем стохастического градиентного спуска
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Создаем функцию потерь
criterion = nn.MSELoss()
epochs = 1
log_interval = 30

# Добавляем тренировочные параметры

f = open('train_input.txt', 'r')
r = open('train_output.txt', 'r')
input_train = []
output_train = []
for i in f:
    input_train += [i]
for i in r:
    output_train += [i]


#запускаем главный тренировочный цикл


for epoch in range(epochs):
   for batch_idx, data, target in zip(range(len(output_train)), input_train, output_train):
       data = data.split(' ')
       target = target.split(' ')
       data = np.array(data[:-1], dtype=np.float32)
       target = np.array(target[:-1], dtype=np.float32)
       data = torch.from_numpy(data)
       target = torch.from_numpy(target)
       data, target = Variable(data), Variable(target)
       optimizer.zero_grad()
       net_out = net(data)
       loss = criterion(net_out, target)
       loss.backward()
       optimizer.step()
       if batch_idx % log_interval == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(output_train),
                          100. * batch_idx / len(output_train), loss.data))

f.close()
r.close()

# Сохраняем НС

PATH = './myNN.pth'
torch.save(net.state_dict(), PATH)