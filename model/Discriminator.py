import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Discriminator(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Discriminator, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.class_strategy = configs.class_strategy
        # self.cnn_seq = nn.Conv1d(configs.seq_len, configs.seq_len, kernel_size=3, padding=1)
        self.cnn_seq = torch.nn.Sequential(
            torch.nn.Conv1d(configs.seq_len, configs.seq_len, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # self.cnn_channel = nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, padding=1)
        self.cnn_channel = torch.nn.Sequential(
            torch.nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.cnn2d_seq = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ])
        self.to_seq_len = torch.nn.Sequential(
            nn.Linear(384, self.seq_len, bias=True),
            torch.nn.PReLU())
        self.dropout = torch.nn.Dropout(p=0.5)
        self.projector = nn.Linear(self.seq_len, 2, bias=True)

        # self.projector = nn.Linear((configs.seq_len-4)*(configs.enc_in-4)*64, 2, bias=True)
        # self.gru = nn.GRU(input_size=configs.d_model, hidden_size=configs.d_model, num_layers=1, batch_first=True)
                # 初始化卷积层参数
        for module in self.cnn_seq.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        for module in self.cnn_channel.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        for module in self.cnn2d_seq.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # 初始化线性层参数
        nn.init.xavier_uniform_(self.to_seq_len[0].weight)
        nn.init.constant_(self.to_seq_len[0].bias, 0.0)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0.0)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(x_enc.shape)
        batch_size = x_enc.shape[0]
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.cnn_seq(x_enc)
        x_enc = self.cnn_channel(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc = x_enc.unsqueeze(1)
        for i in self.cnn2d_seq:
            # print(x_enc.shape)
            x_enc = i(x_enc)
            # print(x_enc.shape)
            # x_enc = F.LeakyReLU(x_enc)
            # x_enc = F.MaxPool2d(x_enc, kernel_size=(2, 1), stride=(2, 1))
        # x_enc = self.cnn2d_seq(x_enc)
        # x_enc = self.gru(x_enc)
        # print(x_enc.shape)
        # 将第四维移动到第二维
        x_enc = x_enc.permute(0, 3, 1, 2)
        # print(x_enc.shape)
        x_enc = nn.Flatten(2, 3)(x_enc)
        # print(x_enc.shape)
        # x_clone = x_enc.clone()
        x_enc = nn.Flatten(0, 1)(x_enc)
        # print(x_enc.shape)
        # tmp = x_enc.view(self.batch_size, self.enc_in, -1)
        # print(x_clone.shape, tmp.shape)
        # print(x_clone == tmp)
        # x_enc = x_enc.view(x_enc.size(0) * x_enc.size(3), -1)
        # print(x_enc.shape)
        x_to_seq_len = self.to_seq_len(x_enc)
        # print(x_to_seq_len.shape)
        x_enc = self.dropout(x_to_seq_len)
        x_enc = self.projector(x_enc)
        # x_enc = nn.Sigmoid()(x_enc)
        # 将第一维拆开
        x_enc = x_enc.view(batch_size, -1, self.enc_in)
        # do bathc norm
        # print(x_enc.shape)
        # x_enc = nn.BatchNorm1d(7)(x_enc)
        # x_enc = x_enc.view(self.batch_size, -1, self.enc_in)
        # print(x_enc.shape)
        # a = b
        x_enc = nn.Sigmoid()(x_enc)
        # print(x_to_seq_len.shape)
        # print(x_to_seq_len.view(batch_size, self.seq_len, -1).shape)
        return x_enc, x_to_seq_len.detach().view(batch_size, self.seq_len, -1)#[:, :, -self.enc_in:]



# class Discriminator(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """

#     def __init__(self, configs):
#         super(Discriminator, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.class_strategy = configs.class_strategy
#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         self.projector = nn.Linear(configs.d_model, 2, bias=True)
#         # self.cnn_seq = nn.Conv1d(configs.seq_len, configs.seq_len, kernel_size=3, padding=1)
#         # self.cnn_channel = nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, padding=1)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev

#         _, _, N = x_enc.shape # B L N
#         # B: batch_size;    E: d_model;
#         # L: seq_len;       S: pred_len;
#         # N: number of variate (tokens), can also includes covariates
#         # x_enc = self.cnn_seq(x_enc)
#         # x_enc = self.cnn_channel(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

#         # Embedding
#         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
#         enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

#         # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
#         # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # B N E -> B N S -> B S N
#         dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             pass

#         return dec_out


#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]