import torch
import torch.nn as nn
from enc_dec import Encoder, Decoder

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.mean = None
        self.std = None

    def forward(self, input, flag):
        if flag=='normalize':
            self.mean = torch.mean(input, dim=0, keepdim=True)
            self.std = (torch.std(input, dim=0, keepdim=True) + 1e-8)
            return (input - self.mean) / self.std
        else:
            return input * self.std + self.mean

class encoder_decoder_small_patch(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, enc_layers=3, dec_layers=3, dropout=0.5, bidirectional=True, lstm_num_layers=2, lstm_hidden_size=8, lstm_resnet=True):
        super(encoder_decoder_small_patch, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.norm = Norm()
        self.encoder = Encoder(input_dim, d_model, layer_num=enc_layers, dropout=dropout, activation=nn.LeakyReLU())
        self.decoder = Decoder(d_model, \
                                output_dim, \
                                layer_num=dec_layers, \
                                dropout=dropout, \
                                activation=nn.Sigmoid(), \
                                bidirectional=bidirectional, \
                                lstm_num_layers=lstm_num_layers, \
                                hidden_size=lstm_hidden_size, \
                                resnet=lstm_resnet)

    def log_density_gaussian(self, sample, mu, logvar):
        '''计算vae的损失函数'''
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return kld_loss

    def normal_sample(self, mean, logvar):
        '''采样'''
        dist = torch.distributions.Normal(0, 1)
        eps = dist.sample(mean.shape).cuda()
        z = mean + torch.exp(.5*logvar) * eps
        return z

    def forward(self, input):
        output = self.norm(input, 'normalize')
        mean, logvar = self.encoder(output)
        z = self.normal_sample(mean, logvar)
        loss_vae = self.log_density_gaussian(z, mean, logvar)
        output = self.decoder(z)
        output = self.norm(output, 'denormalize')
        return output, loss_vae, mean, torch.exp(.5*logvar), z
