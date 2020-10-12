import csv
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import *
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
import time

BATCH_SIZE = 50

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, _ = self.rnn(x, None)  # None represents zero initial hidden state

        out = self.out(r_out[:, -1, :]) # return the last value

        return out

class PridictTM():
    def __init__(self, file_name, k, input_size, hidden_size, num_layers, epoch, LR):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.epoch = epoch
        self.LR = LR
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = RNN(input_size, hidden_size, num_layers)
        # self.rnn.cuda()
        print(self.rnn)

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
            file_name = "../TM_result/GEANT/GRU_OD_pair/\GRU_OD_pair_" + str(i + 1) + ".txt"
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
        OD_list = self.get_OD_list(self.file_name)
        # OD_list = ["OD_1-2"]
        model_path = "../GEANT/model_GRU_OD/"
        # model_path = "/home/liuzhifeng/workspace/code/traffic-matrix-prediction/model_GRU_OD_pair_GEANT/"
        result_list = []
        for i in range(529):
            result_list.append([])

        count = 0
        for OD in OD_list:
            print("Training for ", OD)
            model_name = model_path + "GRU_" + OD + ".pkl"
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


            # reset rnn
            self.rnn = RNN(self.input_size, self.hidden_size, self.num_layers)
            self.rnn.cuda()
            optimizer = torch.optim.Adagrad(self.rnn.parameters(), lr=self.LR)
            loss_func = nn.MSELoss()

            '''
            ################################## train #################################
            if OD.split('_')[1].split('-')[0] == OD.split('_')[1].split('-')[1]:
                continue
            # if os.path.exists(model_name):
            #     continue
            star_time = time.clock()
            for e in range(self.epoch):
                # print("Epoch:", e)

                for step, (batch_x, batch_y) in enumerate(data_loader):
                    # print(batch_x.shape)
                    # print(batch_y.shape)
                    batch_x = batch_x.reshape(BATCH_SIZE, self.k, self.input_size)  # (batch_size, time_step, input_size)
                    batch_y = batch_y.reshape(BATCH_SIZE, self.input_size)
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    prediction = self.rnn.forward(batch_x)
                    loss = loss_func(prediction, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print("Epoch =", e, ", step =", step, ", loss:", loss)

            end_time = time.clock()
            print((end_time - star_time) * 529)

            ################################## train #################################
            # save model
            torch.save(self.rnn.state_dict(), model_name)

            '''
            ################################## test #################################
            
            if OD.split('_')[1].split('-')[0] == OD.split('_')[1].split('-')[1]:
                for i in range(train_len, len(x_data)):
                    result_list[count].append(0)
            else:
            # load model
                self.rnn.load_state_dict(torch.load(model_name))
                star_time = time.clock()
                for i in range(train_len, len(x_data)):
                    test_x = x_data[i].reshape(1, self.k, self.input_size).cuda()
                    test_y = y_data[i].cuda()
                    prediction = self.rnn.forward(test_x).reshape(1).cuda()
                    loss = loss_func(prediction, test_y)

                    # data = []
                    # data.append(str(i - train_len + 1))
                    # data.append(loss.cpu().data.numpy())
                    # self.write_row_to_csv(data, "loss_GRU_OD.csv")

                    prediction_value = prediction.cpu().data.numpy()[0]
                    if prediction_value < 0:
                        prediction_value = -prediction_value
                    result_list[count].append(prediction_value * (max_value - min_value) + min_value)
                end_time = time.clock()
                print((end_time - star_time) / (len(x_data) - train_len) * 529)
            ################################## test #################################

            count += 1

        self.save_TM(result_list)



if __name__ == "__main__":
    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../OD_pair/GEANT-OD_pair_2005-07-26.csv"
    # file_name = "GEANT-OD_pair_2013-03-01.csv"

    k = 10
    input_size = 1
    hidden_size = 50
    num_layers = 1
    epoch = 20
    # 0.065
    LR = 0.065

    predict_tm_model = PridictTM(file_name, k, input_size, hidden_size, num_layers, epoch, LR)
    predict_tm_model.train()


