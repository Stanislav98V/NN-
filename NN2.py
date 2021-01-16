import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import torch.optim as optim
import warnings
warnings.simplefilter('ignore')
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.functional as F


class CommonData():
    """
    Main data louder

    param: trainFile -- path to numpy saved training mfcc features with target phones ID
    param: validFile -- path to numpy saved validation mfcc features with target phones ID
    """

    def __init__(self, train_in, train_out):
        f = open('train_input.txt', 'r')
        r = open('train_output.txt', 'r')
        input_train = []
        output_train = []
        for i in f:
            i = i.split(' ')
            data = np.array(i[:-1], dtype=np.float32)
            data = torch.from_numpy(data)
            input_train += [data]
        for i in r:
            i = i.split(' ')
            target = np.array(i[:-1], dtype=np.float32)
            target = torch.from_numpy(target)
            output_train += [target]
        input_train, output_train = shuffle(input_train, output_train)
        self.train = input_train
        self.train_otv = output_train

    def numBatches(self, batch_size):
        return len(self.train) // batch_size

    def nextBatch(self, batch_size):
        # for i in range(self.numBatches(batch_size)):
        #     batch_xs = torch.tensor(self.train[i * batch_size:(i + 1) * batch_size, -1], dtype=torch.float32)
        #     batch_ys = torch.tensor(self.train_otv[i * batch_size:(i + 1) * batch_size, -1], dtype=torch.float32)
        #
        #     yield batch_xs, batch_ys
        return self.train, self.train_otv


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)  # В i слое 120 узлов, в i+1 84 узла
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 51)

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x


context_len = 2
batch_size = 256

# DNN params:
# input_dim = 13 * (context_len * 2 + 1)  # размерность входных признаков
# ff_dim = 256  # размерность полносвязного слоя
# output_dim = 23  # число таргетов -- фонем

# training network model:
net = MyNet()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# Создаем функцию потерь
criterion = nn.MSELoss()
data = CommonData(1, 2)

loss_list = []

print('TRAINIG IS STARTED...')
for epoch in range(1):
    running_loss = 0
    counter = 0
    cc = 0
    da, targ = data.nextBatch(batch_size)
    for batch_xs, batch_ys in zip(da, targ):

        optimizer.zero_grad()
        output = net(batch_xs)
        loss = criterion(output, batch_ys)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        counter += 1
        if counter % 1000 == 0:
            # print('loss is: {:.4f}'.format((running_loss / counter)))
            # print(cc, loss.item())
            print(running_loss / counter, cc)
            cc += 1
            running_loss = 0
    loss_list.append(running_loss / counter)



path2exp = 'exp'
if not os.path.exists(path2exp):
    os.mkdir(path2exp)

torch.save(net, os.path.join(path2exp, 'MyNet2.pt'))


myModel = torch.load(os.path.join(path2exp, 'MyNet2.pt'))

f = open('test_input.txt', 'r')
r = open('test_output.txt', 'r')
input_test = []
output_test = []
for i in f:
    i = i.split(' ')
    data = np.array(i[:-1], dtype=np.float32)
    input_test += [data]
for i in r:
    i = i.split(' ')
    targ = np.array(i[:-1], dtype=np.float32)
    output_test += [targ]

def count_num(x):
    r = 0
    for i in x:
        if i == 1:
            return r
        r += 1
    return r

with torch.no_grad():
    correct_cnt = 0
    for ftr, true_label in zip(input_test, output_test):
        ftr = torch.from_numpy(ftr).float()
        output = myModel.forward(ftr)
        true_label1 = count_num(true_label)
        correct_cnt += (output.argmax().item() == true_label1)

    print("Frame accuracy is {:.3f}".format(correct_cnt / len(input_test)))