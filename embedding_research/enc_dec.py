import torch
import torch.nn as nn
import numpy as np
from embedding_research.dlinear import series_decomp

class LinearEnc(nn.Module):
    def __init__(self, args):
        super(LinearEnc, self).__init__()
        self.activation = nn.LeakyReLU()
        self.layer_num = args.enc_layers
        self.input_dim = args.seq_len
        self.d_model = args.d_model
        self.structure = args.structure
        self.layer1_dim = args.enc_cnn_layer1_dim
        self.dropout = args.dropout
        self.decompsition = series_decomp(args.moving_avg)

        self.layers = nn.ModuleList()
        self.dims = np.linspace(self.input_dim*2, self.d_model, self.layer_num+1).astype(int)
        for i in range(self.layer_num):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            self.layers.append(self.activation)
            # self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Dropout(self.dropout))

        if self.structure == 'Normal':
            self.linear = nn.Linear(self.d_model, self.d_model)
        elif self.structure == 'VAE':
            self.mean_linear = nn.Linear(self.d_model, self.d_model)
            self.log_var_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, input):
        seasonal_init, trend_init = self.decompsition(input.unsqueeze(-1))
        output = torch.cat([seasonal_init.squeeze(-1), trend_init.squeeze(-1)], dim=-1)
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        if self.structure == 'Normal':
            output = self.linear(output)
            return output
        elif self.structure == 'VAE':
            mean = self.mean_linear(output)
            log_var = self.log_var_linear(output)
            return mean, log_var

class CnnEnc(nn.Module):
    def __init__(self, args):
        super(CnnEnc, self).__init__()
        self.activation = nn.LeakyReLU()
        self.layer_num = args.enc_layers
        self.input_dim = args.seq_len
        self.d_model = args.d_model
        self.structure = args.structure
        self.layer1_dim = args.enc_cnn_layer1_dim
        self.dropout = args.dropout
        self.cnns = nn.ModuleList()
        self.cnns.append(nn.Conv1d(in_channels=1, out_channels=self.layer1_dim, kernel_size=3, stride=1, padding=1))
        # self.cnns.append(nn.Dropout(dropout))
        self.cnns.append(nn.MaxPool1d(kernel_size=2, stride=2))

        channel_num = self.layer1_dim
        for i in range(self.layer_num-1):
            self.cnns.append(nn.Conv1d(in_channels=channel_num, out_channels=channel_num*2, kernel_size=3, stride=1, padding=1))
            # self.cnns.append(nn.Dropout(dropout))
            self.cnns.append(nn.MaxPool1d(kernel_size=2, stride=2))
            channel_num *= 2

        self.flatten = nn.Flatten()
        self.flatten_dim = int(self.layer1_dim * self.input_dim / 2)
        if self.structure == 'Normal':
            self.linear = nn.Linear(self.flatten_dim, self.d_model)
        elif self.structure == 'VAE':
            self.mean_linear = nn.Linear(self.flatten_dim, self.d_model)
            self.log_var_linear = nn.Linear(self.flatten_dim, self.d_model)

    def forward(self, input):
        output = input.unsqueeze(1)
        for i in range(len(self.cnns)):
            output = self.cnns[i](output)

        output = self.flatten(output)
        if self.structure == 'Normal':
            output = self.linear(output)
            return output
        elif self.structure == 'VAE':
            mean = self.mean_linear(output)
            log_var = self.log_var_linear(output)
            return mean, log_var

class LstmDec(nn.Module):
    def __init__(self, args):
        super(LstmDec, self).__init__()
        self.resnet = args.lstm_resnet
        self.layer_num = args.dec_layers
        self.activation = nn.Sigmoid()
        self.rnns = nn.ModuleList()
        self.d_model = args.d_model
        self.output_dim = args.target_len
        self.hidden_size = args.lstm_hidden_size
        self.lstm_num_layers = args.lstm_num_layers
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.rnns.append(nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=self.lstm_num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout))
        for _ in range(self.layer_num-1):
            self.rnns.append(nn.LSTM(input_size=self.hidden_size*(1+self.bidirectional), hidden_size=self.hidden_size, num_layers=self.lstm_num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(self.d_model*self.hidden_size*(1+self.bidirectional), self.output_dim)

    def forward(self, input):
        output = self.rnns[0](input.unsqueeze(-1))[0]
        output = self.activation(output)

        for i in range(self.layer_num-1):
            inner_output = self.rnns[i+1](output)[0]
            inner_output = self.activation(inner_output)
            output = (inner_output + output) if self.resnet else inner_output

        output = self.flatten(output)
        output = self.projection(output)
        return output