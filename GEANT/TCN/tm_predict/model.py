import torch
import torch.nn.functional as F
from torch import nn
from GEANT.TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, k):
        super(TCN, self).__init__()
        # self.dense = nn.Linear(input_size * k, int(input_size / 2) * k)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def sparse(self, inputs):
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[2]
        new_inputs = torch.Tensor([]).cuda()
        # new_inputs = torch.Tensor([])
        # print("new_inputs.shape", new_inputs.shape)

        for i in range(batch_size):
            # new_inputs = torch.cat((new_inputs, inputs[i]), 0)
            temp = inputs[i].reshape(-1)
            # print(temp.shape)
            temp = self.dense(temp).reshape((-1, seq_length))
            new_inputs = torch.cat((new_inputs, temp), 0)

        new_inputs = new_inputs.reshape(batch_size, -1, seq_length)
        return new_inputs
        # print("new_inputs.shape", new_inputs.shape)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # new_inputs = self.sparse(inputs)
        # y1 = self.tcn(new_inputs)  # input should have dimension (N, C, L)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        out = self.linear(y1[:, :, -1])
        out = self.sigmoid(out)
        return out