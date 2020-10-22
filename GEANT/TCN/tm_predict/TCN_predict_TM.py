import csv
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import *
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
import math
from GEANT.TCN.tm_predict.model import TCN
import time

class PridictTM():
    def __init__(self, file_name, k, input_size, input_channel, output_size, channel_sizes, kernel_size,
                 dropout, epochs, lr, USE_CUDA):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.epoch = epochs
        self.LR = lr
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_size = output_size
        self.channel_size = channel_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.USE_CUDA = USE_CUDA

        self.model = TCN(input_size, output_size, channel_sizes, kernel_size=kernel_size, dropout=dropout, k=self.k)
        if self.USE_CUDA:
            self.model = self.model.cuda()

    def read_data(self, file_name):
        df = pd.read_csv(file_name)
        del df["time"]
        data_list = df.values
        # print(data_list)
        # print(data_list.shape)
        # print(type(data_list))

        max_list = np.max(data_list, axis=0)
        min_list = np.min(data_list, axis=0)
        # print(len(max_list))
        # print(len(min_list))

        # OD pair, when O = D, max = min = 0, so data_list will have some nan value
        # change these values to 0
        data_list = (data_list - min_list) / (max_list - min_list)
        data_list[np.isnan(data_list)] = 0

        return data_list, min_list, max_list


    # generate normalized time series data
    # list of ([x1, x2, ..., xk], [xk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data, k):
        x_data = []
        y_data = []
        print("data.shape: ", data.shape)
        length = data.shape[0]
        # print(length)
        for i in range(length - k):
            x = data[i:i+k, :]
            y = data[i+k, :]
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

    # inverse normalization
    def inverse_normalization(self, prediction, y, max_list, min_list):
        inverse_prediction = prediction * (max_list - min_list) + min_list
        inverse_y = y * (max_list - min_list) + min_list

        return inverse_prediction, inverse_y

    # save TM result
    def save_TM(self, TM, file_name):
        f = open(file_name, 'w')
        row, column = TM.shape
        for i in range(row):
            for j in range(column):
                if not TM[i][j] == 0.0:
                    temp = str(i + 1) + ' ' + str(j + 1) + ' ' + str(TM[i][j]) + "\n"
                    f.write(temp)
        f.close()

    def train(self):
        data, min_list, max_list = self.read_data(self.file_name)
        x_data, y_data = self.generate_series(data, self.k)
        print("x_data.shape:", x_data.shape)
        print("y_data.shape:", y_data.shape)
        train_len = int(int(len(x_data) * 0.8) / 50) * 50
        print("Training length: ", train_len)

        data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        loss_func = nn.MSELoss()
        model_name = "TCN_kernel=" + str(self.kernel_size) + "_dropout=" + str(self.dropout) + "_lr=" + str(self.LR) + "_k=" + str(self.k) + ".pkl"


        star_time = time.clock()
        ################################## train ###############################
        for e in range(self.epoch):
            # print("Epoch: ", e)
            result = []
            for step, (batch_x, batch_y) in enumerate(data_loader):
                if USE_CUDA:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    # print("batch_x.shape:", batch_x.shape)
                    # print("batch_y.shape:", batch_y.shape)

                    # TCN input shape: (batch_size, in_channels, seq_length)
                    batch_x = batch_x.reshape(BATCH_SIZE, -1, self.k)
                    prediction = self.model.forward(batch_x).cuda()
                else:
                    batch_x = batch_x
                    batch_y = batch_y
                    # print("batch_x.shape:", batch_x.shape)
                    # print("batch_y.shape:", batch_y.shape)

                    # TCN input shape: (batch_size, in_channels, seq_length)
                    batch_x = batch_x.reshape(BATCH_SIZE, -1, self.k)
                    prediction = self.model.forward(batch_x)
                # print("prediction.shape:", prediction.shape)
                # print(prediction)
                # print(prediction[-1].data.numpy())
                # break
                # result.append(prediction[-1].data.numpy())

                loss = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("Epoch =", e, ", step =", step, ", loss:", loss)
        end_time = time.clock()
        print(end_time - star_time)
        # save model
        torch.save(self.model.state_dict(), model_name)
        
        ################################## train ###############################



        ################################## test ################################
        print("----------------------------test-----------------------\n")
        # load model
        self.model.load_state_dict(torch.load(model_name))
        result = []
        count = 0

        star_time = time.clock()
        for i in range(train_len, len(x_data)):
            test_x = x_data[i].reshape(1, -1, self.k).cuda()
            test_y = y_data[i].reshape(1, self.input_size).cuda()
            # print("test_y.shape:", test_y.shape)
            prediction = self.model.forward(test_x).cuda()
            # print("prediction.shape:", prediction.shape)
            # break
            # loss = loss_func(prediction, test_y)
            # print("Loss for test data " + str(i - train_len + 1) + " is:", loss)

            # save result
            data = []
            # data.append(str(i - train_len + 1))
            # data.append(loss.cpu().data.numpy())
            # self.write_row_to_csv(data, "loss_TCN.csv")

            # inverse normalization
            # inverse_prediction, inverse_y = self.inverse_normalization(prediction.cpu().data.numpy()[0], test_y.cpu().data.numpy()[0], max_list, min_list)
            # inverse_prediction = inverse_prediction.reshape(int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
            # inverse_y = inverse_y.reshape(int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
            #
            # self.save_TM(inverse_prediction, path)
        end_time = time.clock()
        print((end_time - star_time) / (len(x_data) - train_len))
        ################################## test ################################




    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)


if __name__ == "__main__":
    BATCH_SIZE = 50
    USE_CUDA = True
    dropout = 0.05
    clip = -1
    epochs = 20
    kernel_size = 7  # 7
    levels = 8  # 8
    lr = 0.002
    nhid = 25  # 25
    seed = 1111
    input_size = 529
    input_channel = 1
    output_size = 529
    channel_sizes = [nhid] * levels
    k = 10

    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../../../OD_pair/GEANT-OD_pair_2005-07-26.csv"

    predict_tm_model = PridictTM(file_name, k, input_size, input_channel, output_size, channel_sizes, kernel_size,
                                 dropout, epochs, lr, USE_CUDA)
    predict_tm_model.train()



