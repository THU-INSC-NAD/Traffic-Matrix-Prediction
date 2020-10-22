import csv
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import *
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import math
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


BATCH_SIZE = 50

class PridictTM():
    def __init__(self, file_name, k, input_size, output_size, channel_sizes, kernel_size, dropout, LR, epoch):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.epoch = epoch
        self.LR = LR
        self.input_size = input_size
        self.model = TCN(input_size, output_size, channel_sizes, kernel_size=kernel_size, dropout=dropout, k=self.k)
        # self.rnn.cuda()
        # print(self.model)
        self.model.cuda()

    def get_OD_list(self, file_name):
        df = pd.read_csv(file_name)
        OD_list = df.columns.values.tolist()
        del OD_list[0]

        return OD_list

    def read_data(self, file_name, OD):
        df = pd.read_csv(file_name)
        data = np.array(df[OD])

        # min-max normalization

        max_value = np.max(data)
        min_value = np.min(data)
        if not (max_value == 0 and min_value == 0):
            data = (data - min_value) / (max_value - min_value)
        '''

        # z-score
        
        u = np.average(data)
        sigma = np.std(data)
        for i in range(len(data)):
            data[i] = (data[i] - u) / sigma
        '''
        return data, max_value, min_value


    # generate normalized time series data
    # list of ([x1, x2, ..., xk], [xk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data, k):
        x_data = []
        y_data = []
        length = len(data)
        for i in range(length - k):
            x = data[i:i+k]
            y = data[i+k]
            x_data.append(x)
            y_data.append(y)
        x_data = torch.from_numpy(np.array(x_data)).float()
        y_data = torch.from_numpy(np.array(y_data)).float()

        return x_data, y_data

    # generate batch data
    def generate_batch_loader(self, x_data, y_data):
        torch_dataset = Data.TensorDataset(x_data, y_data)
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=True,               # random order data
            num_workers=2,              # multiple threading to read data
        )
        return loader


    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)

    # save TM result
    def save_TM(self, result_list):
        test_data_length = len(result_list[0])
        size = int(math.sqrt(len(result_list)))
        for i in range(test_data_length):
            print("Save TM for data " + str(i))
            file_name = "../../../TM_result/GEANT/\LSTM_OD_pair/\LSTM_OD_pair_" + str(i + 1) + ".txt"
            TM = np.zeros(shape=(size, size))
            row = -1
            column = 0
            for j in range(len(result_list)):
                if j % 23 == 0:
                    row += 1
                    column = 0
                TM[row][column] = result_list[j][i]
                column += 1

            f = open(file_name, 'w')
            for w in range(size):
                for k in range(size):
                    if not TM[w][k] == 0.0:
                        temp = str(w + 1) + ' ' + str(k + 1) + ' ' + str(TM[w][k]) + "\n"
                        f.write(temp)
            f.close()


    def train(self):
        # OD_list = self.get_OD_list(self.file_name)
        OD_list = ["OD_1-2"]
        result_list = []
        for i in range(529):
            result_list.append([])

        count = 0
        for OD in OD_list:
            print("Training for ", OD)
            # print(OD_list)
            data, max_value, min_value = self.read_data(self.file_name, OD)
            x_data, y_data = self.generate_series(data, self.k)
            train_len = int(int(len(x_data) * 0.8) / 50) * 50
            data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

            # min-max
            # for i in range(len(data)):
            #     data[i] = data[i] * (max_value - min_value) + min_value

            # z-score
            # for i in range(len(data)):
            #     data[i] = data[i] * min_value + max_value


            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.LR)
            loss_func = nn.MSELoss()


            ################################## train #################################
            for e in range(self.epoch):
                print("Epoch:", e)
                for step, (batch_x, batch_y) in enumerate(data_loader):
                    # print("origin batch_x.shape:", batch_x.shape)

                    # TCN input shape: (batch_size, in_channels, seq_length)
                    batch_x = batch_x.reshape(BATCH_SIZE, -1, self.k)
                    batch_y = batch_y.reshape(BATCH_SIZE, -1)
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    # print("batch_x.shape:", batch_x.shape)
                    # print("batch_y.shape:", batch_y.shape)
                    prediction = self.model.forward(batch_x)
                    prediction = prediction.cuda()
                    # print("prediction.shape:", prediction.shape)
                    loss = loss_func(prediction, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("Epoch =", e, ", step =", step, ", loss:", loss)

            ################################## train #################################
            # save model
            # torch.save(self.rnn.state_dict(), model_name)

            ################################## test #################################
            # load model
            # self.rnn.load_state_dict(torch.load(model_name))
            for i in range(train_len, len(x_data)):
                test_x = x_data[i].reshape(1, -1, self.k).cuda()
                test_y = y_data[i].cuda()
                prediction = self.model.forward(test_x).reshape(1).cuda()
                loss = loss_func(prediction, test_y)
                print("loss for data", i, ":", loss)
                data = []
                data.append(loss.cpu().data.numpy())
                self.write_row_to_csv(data, "TCN_OD1-2_loss.csv")

                prediction_value = prediction.cpu().data.numpy()[0]
                if prediction_value < 0:
                    prediction_value = -prediction_value
                result_list[count].append(prediction_value * (max_value - min_value) + min_value)
            ################################## test #################################

            count += 1


        # self.save_TM(result_list)



if __name__ == "__main__":
    print(args)

    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../../../OD_pair/GEANT-OD_pair_2005-07-26.csv"

    # best paramaters: hidden_size = 100, LR = 0.065
    BATCH_SIZE = 50
    USE_CUDA = True
    dropout = 0.05
    clip = -1
    epochs = 50
    kernel_size = 7  # 7
    levels = 8  # 8
    lr = 0.065
    nhid = 25  # 25
    seed = 1111
    input_size = 1
    input_channel = 1
    output_size = 1
    channel_sizes = [nhid] * levels
    k = 10

    predict_tm_model = PridictTM(file_name, k, input_size, output_size, channel_sizes, kernel_size, dropout, lr, epochs)
    predict_tm_model.train()

    # for i in range(658):
    #     row = -1
    #     column = 0
    #     for j in range(529):
    #         if j % 23 == 0:
    #             row += 1
    #             column = 0
    #         column += 1
    #         print(row, column)

