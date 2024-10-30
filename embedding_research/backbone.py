import torch
import torch.nn as nn
from enc_dec import *

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.mean = None
        self.std = None

    def forward(self, input, flag):
        if flag=='normalize':
            self.mean = torch.mean(input, dim=1, keepdim=True)
            self.std = (torch.std(input, dim=1, keepdim=True) + 1e-8)
            return (input - self.mean) / self.std
        else:
            return input * self.std + self.mean

class encoder_decoder_small_patch(nn.Module):
    def __init__(self, args):
        super(encoder_decoder_small_patch, self).__init__()
        self.input_dim = args.seq_len
        self.d_model = args.d_model
        self.output_dim = args.seq_len
        self.structure = args.structure
        self.norm = Norm()
        if args.encoder == 'cnn': # optional: 'cnn', 'dlinear'
            self.encoder = CnnEnc(args)
        elif args.encoder == 'dlinear':
            self.encoder = LinearEnc(args)
        else:
            raise ValueError('encoder name not recognized')
        if args.decoder == 'lstm':
            self.decoder = LstmDec(args)
        else:
            raise ValueError('decoder name not recognized')

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

    def ts2z(self, input):
        '''input: [B*N, L], output: [B*N, E]'''
        normalized_input = self.norm(input, 'normalize')
        if self.structure == 'Normal':
            z = self.encoder(normalized_input)
            return z
        elif self.structure == 'VAE':
            mean, logvar = self.encoder(normalized_input)
            z = self.normal_sample(mean, logvar)
            return z

    def z2ts(self, z):
        '''input:[B*N, E], ouput: [B*N, L]'''
        output = self.decoder(z)
        output = self.norm(output, 'denormalize')
        return output

    def forward(self, input, flag='pretrain'):
        if flag == 'pretrain':
            output = self.norm(input, 'normalize')
            if self.structure == 'Normal':
                output = self.encoder(output)
                output = self.decoder(output)
                output = self.norm(output, 'denormalize')
                return output
            elif self.structure == 'VAE':
                mean, logvar = self.encoder(output)
                z = self.normal_sample(mean, logvar)
                loss_vae = self.log_density_gaussian(z, mean, logvar)
                output = self.decoder(z)
                output = self.norm(output, 'denormalize')
                return output, loss_vae, mean, torch.exp(.5*logvar), z
        elif flag == 'ts2z':
            return self.ts2z(input)
        elif flag == 'z2ts':
            return self.z2ts(input)
        else:
            raise ValueError('encoder_decoder_small_patch forward flag not recognized')
