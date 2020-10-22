import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from TCN.tm_predict.model import TCN

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

print(args)

def SeriesGen(N):
    x = torch.arange(1, N, 0.01)
    return torch.sin(x)

# using first k data number to predict the next k + 1 data
def trainDataGen(seq, k):
    dat = list()
    length = len(seq)
    for i in range(length - k):
        x_data = seq[i:i+k]
        y_data = seq[i+k]
        dat.append((x_data, y_data))
    return dat

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)



k = 10
series = SeriesGen(10)
print("series.shape:", series.shape)
data = trainDataGen(series, k)
print("data example: ", data[0])


input_size = 10
output_size = 1
time_step = 10
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_size, output_size, channel_sizes, kernel_size=kernel_size, dropout=args.dropout, k=k)
# print(model)
optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
loss_func = nn.MSELoss()

for epoch in range(5):
    print(epoch)
    count = 0
    result = []
    for seq, y in data[:700]:
        seq = Variable(torch.FloatTensor(seq))
        y = Variable(y)

        # x shape (batch, time_step, input_size)
        seq = seq.reshape((1, -1, k))
        y = y.reshape((1, 1))  # why?


        # TCN input size (batch, in_channels, seq_length)
        prediction = model.forward(seq).reshape(1, 1)
        # print(prediction)
        # print("prediction.shape:", prediction.shape)
        result.append(prediction[-1].data.numpy()[0])
        # print(prediction)
        # print(type(prediction))
        # print(prediction.shape)
        # print(prediction[-1].data.numpy()[0])
        # print(prediction[-1].data.numpy()[0], y[-1].data.numpy()[0])
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1
        if count % 100 == 0:
            print(loss)

fig = plt.figure()
plt.plot(series.numpy())
plt.plot(range(0, 700), result)
plt.show()


'''
##########################test####################################
    result = []
    for seq, y in data[:700]:
        seq = Variable(torch.FloatTensor(seq))
        y = Variable(y)

        # x shape (batch, time_step, input_size)
        seq = seq.reshape((1, 10, 1))
        y = y.reshape((1, 1))  # why?

        prediction = rnn.forward(seq)
        result.append(prediction[-1].data.numpy()[0])
        loss = loss_func(prediction, y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        count += 1
        if count % 100 == 0:
            print(loss)

    fig = plt.figure()
    plt.plot(series.numpy())
    plt.plot(range(0, 700), result)
    plt.show()


print("---------------------------test\n------------------------------")
result = []
# LSTM prediction almost failed. why?
# LSTM, LSTMCell?
# for seq, y in data[700:]:
for seq, y in data[700:]:
    seq = Variable(torch.FloatTensor(seq))
    seq = seq.reshape((1, 10, 1))
    prediction = rnn.forward(seq)
    print(prediction[-1].data.numpy()[0], y)
    result.append(prediction[-1].data.numpy()[0])


fig = plt.figure()
plt.plot(series.numpy())
plt.plot(range(700, 890), result)
plt.show()
'''

