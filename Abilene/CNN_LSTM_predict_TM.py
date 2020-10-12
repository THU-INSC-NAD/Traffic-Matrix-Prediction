import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import *
import numpy as np
import pandas as pd
import csv
import matplotlib as plt
import time

# Hyper Parameters
epoch = 100
BATCH_SIZE = 50
FILE_NAME = "../OD_pair/Abilene-OD_pair_2004-08-01.csv"
LR = 0.01
INPUT_SIZE = 12
HIDDEN_SIZE = 100
NUM_LAYERS = 1
K = 10
USE_CUDA = True


class AlexNet_LSTM(nn.Module):
    def __init__(self):
        super(AlexNet_LSTM, self).__init__()

        ############################### AlexNet CNN #################################
        self.features = nn.Sequential(  # input size (1, 23, 23)
            nn.Conv2d(1, 32, kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout()
        )  # (192, 7, 7)

        self.features2 = nn.Sequential(  # input size (1, 23, 23)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # # nn.Dropout(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # # nn.Dropout(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),

            # nn.Conv2d(128, 128, kernel_size=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout(),

        )

        # CNN to LSTM
        self.middle_out = nn.Linear(128 * 6 * 6, INPUT_SIZE)
        ############################### AlexNet CNN #################################


        ############################## LSTM ###################################
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE * INPUT_SIZE)
        ############################## LSTM ###################################


    def forward(self, x, batch_size):
        x = self.features2(x)
        # print("AlexNet CNN output shape:", x.shape)
        x = x.view(x.size(0), -1)
        middle_output = self.middle_out(x)
        middle_output = middle_output.reshape(batch_size, K, INPUT_SIZE)

        # LSTM
        r_out, (h_n, h_c) = self.rnn(middle_output, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])  # return the last value
        out = self.sigmoid(out)
        return out



# simple CNN with LSTM model
class CNN_LSTM(nn.Module):
    # input image like: (batch_size, 1, input_size, input_size)
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        ############################### CNN ###################################
        self.conv1 = nn.Sequential(  # input size: (1, 12, 12)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, padding=1),  # output size: (16, 8, 8)
            nn.Dropout(),
        )

        # CNN
        self.conv2 = nn.Sequential(  # input size: (16, 12, 12)
            nn.Conv2d(16, 32, 5, 1, 2),  # output size: (32, 12, 12)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output size: (32, 6, 6)
            nn.Dropout()
        )

        # CNN
        self.conv3 = nn.Sequential(  # input size: (32, 6, 6)
            nn.Conv2d(32, 64, 5, 1, 2),  # output size: (64, 6, 6)
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output size: (64, 3, 3)
            nn.Dropout()
        )


        # CNN to LSTM
        self.middle_out = nn.Linear(16 * 8 * 8, INPUT_SIZE)
        ############################### CNN ###################################


        ############################## LSTM ###################################
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE * INPUT_SIZE)
        ############################## LSTM ###################################

    def forward(self, x, batch_size):
        # print("-------------------------------NN forward-------------------------------")
        # CNN
        x = self.conv1(x)
        # print("conv1, x.shape:", x.shape)
        # x = self.conv2(x)
        # print("conv2, x.shape:", x.shape)
        # x = self.conv3(x)
        # print("x.shape:", x.shape)


        x = x.view(x.size(0), -1)  # (batch_size, 16, 8, 8) -> (batch_size, 32 * 6 * 6)
        middle_output = self.middle_out(x)  # (batch_size * K, input_size)
        middle_output = middle_output.reshape(batch_size, K, INPUT_SIZE)  # reshape to (batch_size, time_step, input_size)
        # print("middle_output.shape:", middle_output.shape)
        # LSTM
        r_out, (h_n, h_c) = self.rnn(middle_output, None)  # None represents zero initial hidden state
        # print("r_out.shape:", r_out.shape)
        out = self.out(r_out[:, -1, :])  # return the last value
        # print("r_out.shape:", r_out.shape)
        out = self.sigmoid(out)
        # print("out.shape:", out.shape)
        # print("-------------------------------NN forward-------------------------------")
        return out


class PredictTM():
    def __init__(self):
        self.nn_model = CNN_LSTM()
        self.complex_nn_model = AlexNet_LSTM()

        # GPU
        if USE_CUDA:
            self.nn_model.cuda()
            self.complex_nn_model.cuda()

    def read_data(self):
        df = pd.read_csv(FILE_NAME)
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

        # change to TM list
        data = []
        for i in range(data_list.shape[0]):
            data.append(data_list[i].reshape(INPUT_SIZE, INPUT_SIZE))
        data = np.array(data)

        return data, min_list, max_list

    # generate normalized time series data
    # list of ([TM1, TM2, TM3, .. TMk], [TMk+1])
    # using first k data to predict the k+1 data
    def generate_series(self, data):
        length = data.shape[0]
        x_data = []
        y_data = []
        for i in range(length - K):
            x = data[i:i+K]
            y = data[i+K]
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
        data, min_list, max_list = self.read_data()
        x_data, y_data = self.generate_series(data)
        print("x_data.shape:", x_data.shape)
        print("y_data.shape:", y_data.shape)
        train_len = int(int(len(x_data) * 0.8) / 50) * 50
        print("training len:", train_len)
        data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

        # print(self.nn_model)
        optimizer = torch.optim.Adagrad(params=self.nn_model.parameters(), lr=LR)
        # optimizer = torch.optim.Adagrad(params=self.complex_nn_model.parameters(), lr=LR)
        loss_func = nn.MSELoss()

        '''
        star_time = time.clock()
        #################################### train ####################################
        for e in range(EPOCH):
            # print("Epoch:", e)
            for step, (batch_x, batch_y) in enumerate(data_loader):
                # print("batch_x.shape:", batch_x.shape)
                # print("batch_y.shape", batch_y.shape)

                # GPU
                if USE_CUDA:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    prediction = self.nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE).cuda()
                    # prediction = self.complex_nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE).cuda()
                else:
                    # prediction = self.nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE)
                    prediction = self.complex_nn_model.forward(batch_x.reshape(BATCH_SIZE*K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=BATCH_SIZE)

                # break
                # print(prediction.shape)
                batch_y = batch_y.reshape(BATCH_SIZE, INPUT_SIZE * INPUT_SIZE)
                loss = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("Epoch =", e, ", step =", step, ", loss:", loss)
        end_time = time.clock()
        print(end_time - star_time)
        #################################### train ####################################
        '''
        # save model
        model_name = "CNN_LSTM_LR=" + str(LR) + "_hidden=" + str(HIDDEN_SIZE) + ".pkl"
        # model_name = "complex_CNN_LSTM_LR=" + str(LR) + "_hidden=" + str(HIDDEN_SIZE) + ".pkl"
        # torch.save(self.nn_model.state_dict(), model_name)
        # torch.save(self.complex_nn_model.state_dict(), model_name)


        #################################### test ####################################
        print("----------------------------test-----------------------\n")

        # load model
        self.nn_model.load_state_dict(torch.load(model_name))
        # self.complex_nn_model.load_state_dict(torch.load(model_name))
        star_time = time.clock()
        for i in range(train_len, len(x_data)):
            test_x = x_data[i]
            test_y = y_data[i]
            # print("test_x.shape:", test_x.shape)
            # print("test_y.shape", test_y.shape)

            # GPU
            if USE_CUDA:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                prediction = self.nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1).cuda()
                # prediction = self.complex_nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1).cuda()
            else:
                # prediction = self.nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1)
                prediction = self.complex_nn_model.forward(test_x.reshape(K, 1, INPUT_SIZE, INPUT_SIZE), batch_size=1)

            # print("prediction.shape:", prediction.shape)


            test_y = test_y.reshape(1, INPUT_SIZE * INPUT_SIZE)

            # loss = loss_func(prediction, test_y)
            # print("Loss for test data " + str(i - train_len + 1) + " is:", loss)
            # # save result
            '''
            data = []
            data.append(str(i - 5049))
            if USE_CUDA:
                data.append(loss.cpu().data.numpy())
            else:
                data.append(loss.data.numpy())
            self.write_row_to_csv(data, "loss_CNN_LSTM.csv")
            '''
            # inverse normalization
            if USE_CUDA:
                # print(prediction.cpu().data.numpy().shape)
                prediction = prediction.cpu().data.numpy()[0]
                test_y = test_y.cpu().data.numpy()[0]
                inverse_prediction, inverse_y = self.inverse_normalization(prediction, test_y, max_list, min_list)
                inverse_prediction = inverse_prediction.reshape(INPUT_SIZE, INPUT_SIZE)
                path = "../TM_result/Abilene/CNN_LSTM/CNN_LSTM_" + str(i - train_len + 1) + ".txt"
                # self.save_TM(inverse_prediction, path)

            else:
                # print(prediction.data.numpy().shape)
                prediction = prediction.data.numpy()[0]
                test_y = test_y.data.numpy()[0]
                inverse_prediction, inverse_y = self.inverse_normalization(prediction, test_y, max_list, min_list)
                inverse_prediction = inverse_prediction.reshape(INPUT_SIZE, INPUT_SIZE)
                path = "../TM_result/Abilene/CNN_LSTM/CNN_LSTM_" + str(i - train_len + 1) + ".txt"
                # self.save_TM(inverse_prediction, path)

        #################################### test ####################################
        end_time = time.clock()
        print((end_time - star_time) / (len(x_data) - train_len))

    def write_row_to_csv(self, data, file_name):
        with open(file_name, 'a+', newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerow(data)

if __name__ == "__main__":
    predict_tm_model = PredictTM()
    predict_tm_model.train()


    # np_list = []
    # for i in range(54):
    #     if i % 3 == 0:
    #         np_list.append(np.array([i-3, i-2, i-1]))
    # np_list = np.array(np_list)
    # print(np_list)
    # print(np_list.shape)
    # print("-----------------------")
    # np_list = np_list.reshape(3, 2, 3, 3)
    # print(np_list)
    # print("-----------------------")
    # np_list = np_list.reshape(6, 1, 3, 3)
    # print(np_list)
    # print(np_list.shape)