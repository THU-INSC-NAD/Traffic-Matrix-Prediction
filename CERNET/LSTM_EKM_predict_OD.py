import csv
import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
import time
import random
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
        # print(x)
        x_traffic = x[:, 0:self.time_step, :]
        x_week_day = x[:, -2, :].long()
        x_hour = x[:, -1, :].long()
        # print(x_traffic.shape, x_week_day.shape, x_hour.shape)
        # print(x_traffic, x_week_day, x_hour)

        # embed hour and week_day data
        x_week_day_embed = self.week_day_embeds(x_week_day).squeeze(1)
        x_hour_embed = self.hour_embeds(x_hour).squeeze(1)


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

    def get_OD_list(self, file_name):
        df = pd.read_csv(file_name)
        OD_list = df.columns.values.tolist()
        del OD_list[0]
        del OD_list[-1]
        del OD_list[-1]

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
    # list of ([x1, x2, ..., xk, hour_k+1, week_day_k+1], [xk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data, week_day_data, hour_data, k):
        x_data = []
        y_data = []
        length = len(data)
        data = data.tolist()
        week_day_data = week_day_data.tolist()
        hour_data = hour_data.tolist()
        for i in range(length - k):
            x = data[i: i + k]
            x.append(week_day_data[i + k])
            x.append(hour_data[i + k])
            x = np.array(x)
            y = data[i + k]
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
            batch_size=self.BATCH_SIZE,      # mini batch size
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
            file_path = "../TM_result/CERNET/LSTM-EKM_OD_pair/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = file_path + "LSTM-EKM_OD_pair_" + str(i + 1) + ".txt"
            TM = np.zeros(shape=(size, size))
            row = -1
            column = 0
            for j in range(len(result_list)):
                if j % self.node_num == 0:
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
        # print(OD_list)
        # OD_list = ["OD_1-14"]

        model_path = "../CERNET/model_LSTM-EKM_OD/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        result_list = []
        for i in range(self.node_num * self.node_num):
            result_list.append([])

        # get week day and hour data
        df = pd.read_csv(file_name)
        week_day_data = np.array(df["week_day"])
        hour_data = np.array(df["hour"])


        count = 0

        # 只有几个合法数值的 OD 对，矫正为 0
        # 否则 cluster 报错，聚类个数太少
        zero_OD_list = ["OD_4-2", "OD_4-5", "OD_4-9", "OD_4-11", "OD_4-13", "OD_4-14", "OD_5-9", "OD_5-14",
                        "OD_9-2", "OD_9-4", "OD_9-5", "OD_9-6", "OD_9-11", "OD_9-13", "OD_9-14"]

        for OD in OD_list:
            print("Training for ", OD)
            model_name = model_path + "LSTM-EKM_" + OD + ".pkl"
            # print(OD_list)

            # get traffic data
            traffic_data, max_traffic, min_traffic = self.read_data(self.file_name, OD)

            # generate data series
            traffic_data_series, y_data_series = self.generate_series(traffic_data, week_day_data, hour_data, self.k)
            # print(traffic_data_series[0], y_data_series[0])

            # get train_data and test_data
            train_len = int(int(len(traffic_data_series) * 0.8) / BATCH_SIZE) * BATCH_SIZE

            x_train = traffic_data_series[:train_len]
            y_train = y_data_series[:train_len]
            x_test = traffic_data_series[train_len:]
            y_test = y_data_series[train_len:]

            train_data_loader = self.generate_batch_loader(x_train, y_train)

            # reset rnn
            self.model = EmbedRNN(traffic_dim, hour_embed_dim, week_day_embed_dim,
                                  rnn_hidden_size, rnn_num_layers, k)
            self.model.cuda()

            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.LR)
            loss_func = nn.MSELoss()

            '''
            ################################## train #################################
            # 全 0 的列，没有流量，不预测
            if max_traffic == 0 and min_traffic == 0:
                continue
            if os.path.exists(model_name):
                continue
            if OD in zero_OD_list:
                continue
            star_time = time.clock()

            for e in range(self.epoch):
                print("Epoch:", e)

                for step, (batch_x, batch_y) in enumerate(train_data_loader):
                    batch_x = batch_x.unsqueeze(2).cuda()
                    batch_y = batch_y.unsqueeze(1).cuda()
                    # print("batch_x.shape", batch_x.shape)
                    # print("batch_y.shape", batch_y.shape)

                    prediction = self.model.forward(batch_x)
                    loss = loss_func(prediction, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print("Epoch =", e, ", step =", step, ", loss:", loss)
            end_time = time.clock()
            print("training time", (end_time - star_time))
            ################################## train #################################
            # save model
            torch.save(self.model.state_dict(), model_name)
            '''

            ################################## test #################################
            if (max_traffic == 0 and min_traffic == 0) or OD in zero_OD_list:
                for i in range(len(x_test)):
                    result_list[count].append(0)
            else:
                # k-means cluster
                # 每个OD对，Canopy有点慢了。直接按照 timestep设置吧
                self.cluster_number = int(24 * (60 / self.time_step))
                # 全 0 列，cluster number = 24 * 60 / time_step 有问题
                if max_traffic > 0:
                    cluster_data = traffic_data[:train_len].reshape(-1, 1)
                    # print(cluster_data)
                    kmeans_cls = KMeans(self.cluster_number)
                    kmeans_cls.fit(cluster_data)
                    centroids = kmeans_cls.cluster_centers_
                    # print(centroids)
                    
                # load model
                self.model.load_state_dict(torch.load(model_name))
                out_file = "./compare_EKM/LSTM-EKM_" + OD + ".csv"
                star_time = time.clock()
                for i in range(len(x_test)):
                    batch_x = x_test[i].cuda().unsqueeze(0).unsqueeze(2)
                    batch_y = y_test[i].cuda().unsqueeze(0).unsqueeze(1)

                    prediction = self.model.forward(batch_x)
                    loss = loss_func(prediction, batch_y)
                    # loss = loss_func(prediction, test_y)

                    # data = []
                    # data.append(str(i - train_len + 1))
                    # data.append(loss.data.numpy())
                    # self.write_row_to_csv(data, "loss_LSTM_OD.csv")
    
                    prediction_value = prediction.cpu().data.numpy()[0][0]

                    # find centroid by the predicted traffic
                    # prediction_value = prediction_value + \
                    #                    centroids[kmeans_cls.predict(prediction.cpu().data.numpy())][0][0]

                    # find centroid by the previous traffic
                    if prediction_value < 0:
                        prediction_value = -prediction_value

                    prediction_value = prediction_value + centroids[kmeans_cls.
                        predict(x_test[i][self.k - 1].data.numpy().reshape(-1, 1))][0][0]

                    prediction_value /= 2.0

                    prediction_traffic = prediction_value * (max_traffic - min_traffic) + min_traffic
                    origin_traffic = batch_y.cpu().data.numpy()[0][0] * (max_traffic - min_traffic) + min_traffic

                    # data = []
                    # data.append(origin_traffic)
                    # data.append(prediction_traffic)
                    # data.append(abs(prediction_traffic - origin_traffic) / origin_traffic)
                    # self.write_row_to_csv(data, out_file)

                    result_list[count].append(prediction_traffic)

                end_time = time.clock()
                print("average prediction time:", (end_time - star_time) / len(x_test) * self.node_num * self.node_num)
            ################################## test #################################
            
            count += 1

        self.save_TM(result_list)



if __name__ == "__main__":
    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../OD_pair/CERNET-OD_pair_embed_2013-03-01.csv"
    # file_name = "CERNET-OD_pair_2013-03-01.csv"

    k = 10
    traffic_dim = 1
    week_day_embed_dim = 100
    hour_embed_dim = 100
    rnn_hidden_size = 200
    rnn_num_layers = 1
    epoch = 100
    LR = 0.01
    time_step = 5

    node_num = 14

    # hidden_size = 30
    # epoch = 20
    # 0.065
    # LR = 0.065


    # def __init__(self, file_name, k, traffic_dim, hour_embed_dim, week_day_embed_dim,
    #              rnn_hidden_size, rnn_num_layers, epoch, LR, BATCH_SIZE, time_step):

    predict_tm_model = PridictTM(file_name, k, traffic_dim, hour_embed_dim, week_day_embed_dim,
                                 rnn_hidden_size, rnn_num_layers, epoch, LR, BATCH_SIZE, time_step, node_num)
    predict_tm_model.train()


