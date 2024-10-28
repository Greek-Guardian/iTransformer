import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, structure='VAE', layer_num=3, dropout=0.5, activation=nn.LeakyReLU()):
        super(Encoder, self).__init__()
        self.activation = activation
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.d_model = d_model
        self.structure = structure
        self.layer1_dim = 16
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
            self.linear = nn.Linear(self.flatten_dim, d_model)
        elif self.structure == 'VAE':
            self.mean_linear = nn.Linear(self.flatten_dim, d_model)
            self.log_var_linear = nn.Linear(self.flatten_dim, d_model)

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

class Decoder(nn.Module):
    def __init__(self, d_model, output_dim, layer_num=3, dropout=0.5, activation=nn.Sigmoid(), bidirectional=True, lstm_num_layers=2, hidden_size=8, resnet=True):
        super(Decoder, self).__init__()
        self.resnet = resnet
        self.layer_num = layer_num
        self.activation = activation
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout))
        for _ in range(self.layer_num-1):
            self.rnns.append(nn.LSTM(input_size=hidden_size*(1+bidirectional), hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(d_model*hidden_size*(1+bidirectional), output_dim)

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