import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from CERNET.DBN.RBM import RBM
import numpy as np
import pandas as pd
import csv
import os
import math
import time
BATCH_SIZE = 50


class DBN(nn.Module):
    def __init__(self,
                visible_units = 10,
                hidden_units = [100, 100, 100, 100, 100, 100, 100, 100],
                k = 2,
                learning_rate = 0.001,
                learning_rate_decay = False,
                xavier_init = False,
                increase_to_cd_k = False,
                use_gpu = False
                ):
        super(DBN,self).__init__()

        self.n_layers = len(hidden_units)
        self.rbm_layers =[]
        self.rbm_nodes = []
        self.out_layer = nn.Linear(100, 1)

        # Creating different RBM layers
        for i in range(self.n_layers ):
            input_size = 0
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i-1]
            rbm = RBM(visible_units = input_size,
                    hidden_units = hidden_units[i],
                    k= k,
                    learning_rate = learning_rate,
                    learning_rate_decay = learning_rate_decay,
                    xavier_init = xavier_init,
                    increase_to_cd_k = increase_to_cd_k,
                    use_gpu=use_gpu)

            self.rbm_layers.append(rbm)

        # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)

        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])


    def forward(self , input_data):
        '''
            running the forward pass
            do not confuse with training this just runs a foward pass
        '''
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0] , -1)).type(torch.FloatTensor)#flatten
            p_v,v = self.rbm_layers[i].to_hidden(v)
        result = self.out_layer(v)
        # return p_v,v
        return result

    def reconstruct(self,input_data):
        '''
        go till the final layer and then reconstruct
        '''
        h = input_data
        p_h = 0
        for i in range(len(self.rbm_layers)):
            h = h.view((h.shape[0] , -1)).type(torch.FloatTensor)#flatten
            p_h,h = self.rbm_layers[i].to_hidden(h)

        v = h
        for i in range(len(self.rbm_layers)-1,-1,-1):
            v = v.view((v.shape[0] , -1)).type(torch.FloatTensor)
            p_v,v = self.rbm_layers[i].to_visible(v)
        return p_v,v



    def train_static(self, train_data,train_labels,num_epochs=50,batch_size=10):
        '''
        Greedy Layer By Layer training
        Keeping previous layers as static
        '''

        tmp = train_data

        for i in range(len(self.rbm_layers)):
            print("-"*20)
            print("Training the {} st rbm layer".format(i+1))

            tensor_x = tmp.type(torch.FloatTensor) # transform to torch tensors
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
            _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=batch_size,drop_last = True) # create your dataloader

            self.rbm_layers[i].train(_dataloader , num_epochs,batch_size)
            # print(train_data.shape)
            v = tmp.view((tmp.shape[0] , -1)).type(torch.FloatTensor)#flatten
            p_v , v = self.rbm_layers[i].forward(v)
            tmp = v
            # print(v.shape)
        return

    def train_ith(self, train_data,train_labels,num_epochs,batch_size,ith_layer):
        '''
        taking ith layer at once
        can be used for fine tuning
        '''
        if(ith_layer-1>len(self.rbm_layers) or ith_layer<=0):
            print("Layer index out of range")
            return
        ith_layer = ith_layer-1
        v = train_data.view((train_data.shape[0] , -1)).type(torch.FloatTensor)

        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].forward(v)

        tmp = v
        tensor_x = tmp.type(torch.FloatTensor) # transform to torch tensors
        tensor_y = train_labels.type(torch.FloatTensor)
        _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
        _dataloader = torch.utils.data.DataLoader(_dataset , batch_size=batch_size,drop_last=True)
        self.rbm_layers[ith_layer].train(_dataloader, num_epochs,batch_size)
        return

class PridictTM():
    def __init__(self, file_name, k, epochs, LR):
        # super(PridictTM, self).__init__()
        self.file_name = file_name
        self.k = k
        self.dbn = DBN()
        self.epoch = epochs
        self.LR = LR
        # self.rnn.cuda()

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
            x = data[i:i + k]
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
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,  # random order data
            num_workers=2,  # multiple threading to read data
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
            file_name = "../../TM_result/CERNET/\DBN/\DBN_" + str(i + 1) + ".txt"
            TM = np.zeros(shape=(size, size))
            row = -1
            column = 0
            for j in range(len(result_list)):
                if j % 14 == 0:
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
        model_path = "../../CERNET/model_DBN_OD/"
        result_list = []
        for i in range(196):
            result_list.append([])

        count = 0
        for OD in OD_list:
            print("Training for ", OD)
            model_name = model_path + "DBN_" + OD + ".pkl"

            # print(OD_list)
            data, max_value, min_value = self.read_data(self.file_name, OD)
            x_data, y_data = self.generate_series(data, self.k)
            train_len = int(int(len(x_data) * 0.8) / 50) * 50
            data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])

            # reset dbn
            self.dbn = DBN()

            optimizer = torch.optim.Adagrad(self.dbn.parameters(), lr=self.LR)
            loss_func = nn.MSELoss()

            '''
            ################################## train #################################
            # if os.path.exists(model_name):
            #     continue
            if OD.split('_')[1].split('-')[0] == OD.split('_')[1].split('-')[1]:
                continue
            star_time = time.clock()
            for e in range(self.epoch):
                # print("Epoch:", e)
                for step, (batch_x, batch_y) in enumerate(data_loader):
                    # batch_x = batch_x.reshape(BATCH_SIZE, self.k, self.input_size)  # (batch_size, time_step, input_size)
                    # batch_y = batch_y.reshape(BATCH_SIZE, self.input_size)
                    # batch_x = batch_x.cuda()
                    # batch_y = batch_y.cuda()
                    prediction = self.dbn.forward(batch_x).reshape(BATCH_SIZE)
                    loss = loss_func(prediction, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # print("Epoch =", e, ", step =", step, ", loss:", loss)

            ################################## train #################################
            # save model
            torch.save(self.dbn.state_dict(), model_name)
            end_time = time.clock()
            print((end_time - star_time) * 196)
            '''


            ################################## test #################################
            if OD.split('_')[1].split('-')[0] == OD.split('_')[1].split('-')[1]:
                for i in range(train_len, len(x_data)):
                    result_list[count].append(0)
            else:
                # load model
                self.dbn.load_state_dict(torch.load(model_name))
                star_time = time.clock()
                for i in range(train_len, len(x_data)):
                    test_x = x_data[i].reshape(1, -1)
                    test_y = y_data[i]
                    prediction = self.dbn.forward(test_x).reshape(1)
                    # loss = loss_func(prediction, test_y)
                    # data = []
                    # data.append(loss.data.numpy())
                    # self.write_row_to_csv(data, "DBN_OD1-2_loss.csv")
    
                    prediction_value = prediction.data.numpy()[0]
                    if prediction_value < 0:
                        prediction_value = -prediction_value
                    result_list[count].append(prediction_value * (max_value - min_value) + min_value)
                end_time = time.clock()
                print((end_time - star_time) / (len(x_data) - train_len) * 196)
            ################################## test #################################
            count += 1


        self.save_TM(result_list)

if __name__ == "__main__":
    # PridictTM (self, file_name, k, input_size, hidden_size, num_layers)
    file_name = "../../OD_pair/CERNET-OD_pair_2013-03-01.csv"

    # best paramaters: hidden_size = 100, LR = 0.065
    k = 10
    epoch = 20
    # 0.065
    LR = 0.065

    predict_tm_model = PridictTM(file_name, k, epoch, LR)
    predict_tm_model.train()