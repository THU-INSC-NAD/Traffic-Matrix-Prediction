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
import time
import os
from sklearn.cluster import KMeans

BATCH_SIZE = 50

class EmbedRNN(nn.Module):
    '''
    combine LSTM with hour and day embedding
    :param traffic_dim: the dimension of traffic
    :param hour_embed_dim: the dimension of hour data embedding
    :param week_day_embed_dim: the dimension of week_day data embedding
    :param rnn_input_size: the input_size of LSTM
    :param rnn_hidden_size: the hidden size of LSTM
    :param rnn_num_layer: the number of hidden layers of LSTM
    '''
    def __init__(self, traffic_dim, hour_embed_dim, week_day_embed_dim,
                 rnn_hidden_size, rnn_num_layers, k):
        super(EmbedRNN, self).__init__()
        self.traffic_dim = traffic_dim
        self.time_step = k

        # hour and week_day information embedding
        self.hour_embeds = nn.Embedding(24, hour_embed_dim)
        self.week_day_embeds = nn.Embedding(7, week_day_embed_dim)

        # RNN model
        self.rnn = nn.LSTM(
            input_size=traffic_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True
        )

        self.out = nn.Linear(rnn_hidden_size + week_day_embed_dim + hour_embed_dim, traffic_dim)


    def forward(self, x):
        # 对输入 x 进行拆分，分成 traffic、hour、week_day
        # hour 过 hour_embedding，week_day 过 week_day_embedding
        # traffic 过 LSTM，最后一个隐藏层再和两个 embedding 拼接，然后过线性层
        x_traffic = x[:, :, :self.traffic_dim]
        x_week_day = x[:, -1, -2].long()
        x_hour = x[:, -1, -1].long()
        # print(x_traffic.shape, x_week_day.shape, x_hour.shape)
        # print(x_traffic, x_week_day, x_hour)

        # embed hour and week_day data
        x_week_day_embed = self.week_day_embeds(x_week_day).squeeze(1)
        x_hour_embed = self.hour_embeds(x_hour).squeeze(1)
        # print(x_week_day_embed.shape, x_hour_embed.shape)

        # LSTM forward
        # None represents zero initial hidden state
        r_out, (h_n, h_c) = self.rnn(x_traffic, None)
        # print("r_out.shape:", r_out.shape)
        # print("r_out[:, -1, :].shape:", r_out[:, -1, :].shape)

        out_input = torch.cat((r_out[:, -1, :], x_week_day_embed, x_hour_embed), 1)
        # print("out_input.shape:", out_input.shape)


        # return the last
        out = self.out(out_input)

        return out

class PridictTM():
    def __init__(self, file_name, k, traffic_dim, hour_embed_dim, week_day_embed_dim,
                rnn_hidden_size, rnn_num_layers, epoch, LR, BATCH_SIZE, time_step, node_num):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.epoch = epoch
        self.LR = LR
        self.traffic_dim = traffic_dim
        self.hour_embed_dim = hour_embed_dim
        self.week_day_embed_dim = week_day_embed_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.BATCH_SIZE = BATCH_SIZE
        self.time_step = time_step
        self.cluster_num = 0
        self.node_num = node_num

        self.model = EmbedRNN(traffic_dim, hour_embed_dim, week_day_embed_dim,
                              rnn_hidden_size, rnn_num_layers, k)
        self.model.cuda()
        print(self.model)

    def read_data(self, file_name):
        df = pd.read_csv(file_name)
        week_day_data = df["week_day"]
        hour_data = df["hour"]

        del df["time"]

        data_list = df.values
        # print(data_list)
        # print(data_list.shape)
        # print(data_list[0].shape)
        # print(data_list[:, 0].shape)
        # print(type(data_list))

        max_list = np.max(data_list, axis=0)
        min_list = np.min(data_list, axis=0)
        max_list = max_list[:-2]
        min_list = min_list[:-2]

        for i in range(max_list.shape[0]):
            if not max_list[i] - min_list[i] == 0:
                data_list[:, i] = (data_list[:, i] - min_list[i]) / (max_list[i] - min_list[i])

        return data_list, min_list, max_list, week_day_data, hour_data


    # generate normalized time series data
    # list of ([x1, x2, ..., xk], [xk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data, k):
        x_data = []
        y_data = []
        length = data.shape[0]
        y_length = data.shape[1]
        # print(length)
        for i in range(length - k):
            x = data[i: i + k, :]
            y = data[i + k, :]

            # y 数据不需要带有 week 和 hour
            y = y[:y_length - 2]

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
    def inverse_normalization(self, prediction, max_traffic_list, min_traffic_list):
        for i in range(prediction.shape[0]):
            if prediction[i] < 0:
                prediction[i] = -prediction[i]
            prediction[i] = prediction[i] * (max_traffic_list[i] - min_traffic_list[i]) + max_traffic_list[i]

        return prediction

        # inverse_y = y * (max_list - min_list) + min_list
        # return inverse_prediction, inverse_y

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
        # read traffic data
        # traffic_data: [6048, 198], 198 = 196 + 2，TM 压成一行 196 + week + hour
        traffic_data, min_traffic_list, max_traffic_list, week_day_data, hour_data = self.read_data(self.file_name)

        # generate data series
        traffic_data_series, y_data_series = self.generate_series(traffic_data, self.k)
        # print(traffic_data_series.shape)
        # print(y_data_series.shape)

        # get train_data and test_data
        train_len = int(int(len(traffic_data_series) * 0.8) / BATCH_SIZE) * BATCH_SIZE

        x_train = traffic_data_series[:train_len]
        y_train = y_data_series[:train_len]
        x_test = traffic_data_series[train_len:]
        y_test = y_data_series[train_len:]


        train_data_loader = self.generate_batch_loader(x_train, y_train)

        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.LR)
        loss_func = nn.MSELoss()
        model_name = "LSTM-EKM_TM_input=" + str(self.traffic_dim) + "_hidden=" + str(self.rnn_hidden_size) + \
                     "_k=" + str(self.k) + ".pkl"


        star_time = time.clock()
        '''
        ################################## train ###############################
        for e in range(self.epoch):
            print("Epoch: ", e)
            for step, (batch_x, batch_y) in enumerate(train_data_loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                # print("batch_x.shape:", batch_x.shape)
                # print("batch_y.shape:", batch_y.shape)
                prediction = self.model.forward(batch_x).cuda()
                # print("prediction.shape:", prediction.shape)
                # print(prediction)
                # print(prediction[-1].data.numpy())

                loss = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("Epoch =", e, ", step =", step, ", loss:", loss)
        end_time = time.clock()
        print("Training time:", end_time - star_time)
        
        # save model 
        torch.save(self.model.state_dict(), model_name)
        ################################## train ###############################
        '''

        ################################## test ################################
        path = "../TM_result/CERNET/LSTM-EKM_TM/"
        # load model
        self.model.load_state_dict(torch.load(model_name))
        # print(x_test[0][self.k - 1][:-2].data.numpy().shape,
        #       x_test[0][self.k - 1][:-2].data.numpy().shape)

        # k-means cluster
        cluster_data = traffic_data[:train_len, :self.node_num * self.node_num]
        self.cluster_number = int(24 * (60 / self.time_step))
        kmeans_cls = KMeans(self.cluster_number)
        kmeans_cls.fit(cluster_data)
        centroids = kmeans_cls.cluster_centers_
        # print(cluster_data.shape)
        # print(len(centroids))
        # print(centroids[kmeans_cls.predict(cluster_data[0].reshape(1, -1))])
        # print("-------")
        # print(centroids[kmeans_cls.predict(x_test[5][self.k - 1][:-2].data.numpy().reshape(1, -1))])
        # print("-------")
        # print(centroids[kmeans_cls.
        #       predict(x_test[0][self.k - 1][:-2].data.numpy().reshape(1, -1))][0])

        star_time = time.clock()
        for i in range(len(x_test)):
            test_x = x_test[i].cuda().unsqueeze(0)
            test_y = y_test[i].cuda().unsqueeze(0)
            # print(test_x.shape, test_y.shape)
            # print("test_y.shape:", test_y.shape)
            prediction = self.model.forward(test_x).cuda()
            # print("prediction.shape:", prediction.shape)

            # get prediction loss
            # loss = loss_func(prediction, test_y)
            # print("Loss for test data " + str(i - train_len + 1) + " is:", loss)
            # break
            # save result
            # data = []
            # data.append(str(i - train_len + 1))
            # data.append(loss.cpu().data.numpy())
            # self.write_row_to_csv(data, "loss_LSTM.csv")

            # inverse normalization
            # print(prediction.cpu().data.numpy()[0])
            # print(prediction.cpu().data.numpy()[0].shape)
            # print(max_traffic_list.shape)
            # print(min_traffic_list.shape)
            inverse_prediction = self.inverse_normalization(prediction.cpu().data.numpy()[0],
                                                            max_traffic_list, min_traffic_list)

            # find centroid by the previous traffic
            prediction_value = (inverse_prediction +
                                centroids[kmeans_cls.predict
                                (x_test[i][self.k - 1][:-2].data.numpy().reshape(1, -1))][0]) / 2.0

            # 矩阵聚类有可能出现负数，避免这一情况影响 TE
            for j in range(prediction_value.shape[0]):
                if prediction_value[j] < 0:
                    prediction_value[j] = -prediction_value[j]


            prediction_value = inverse_prediction
            prediction_value = prediction_value.reshape(self.node_num, self.node_num)

            if not os.path.exists(path):
                os.makedirs(path)
            our_file = path + "LSTM-EKM_TM_" + str(i + 1) + ".txt"
            self.save_TM(prediction_value, our_file)

        ################################## test ################################
        end_time = time.clock()
        print("test time:", (end_time - star_time) / len(x_test))


    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)


if __name__ == "__main__":
    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../OD_pair/CERNET-OD_pair_embed_2013-03-01.csv"
    k = 10
    traffic_dim = 196
    week_day_embed_dim = 100
    hour_embed_dim = 100
    rnn_hidden_size = 200
    rnn_num_layers = 1
    epoch = 200
    LR = 0.01
    time_step = 5

    node_num = 14

    predict_tm_model = PridictTM(file_name, k, traffic_dim, hour_embed_dim, week_day_embed_dim,
                                 rnn_hidden_size, rnn_num_layers, epoch, LR, BATCH_SIZE, time_step, node_num)
    predict_tm_model.train()




