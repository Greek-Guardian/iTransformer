import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from backbone import encoder_decoder_small_patch

class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.train_strategy = configs.train_strategy # optionlal: 'z2z', 'x2y'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.class_strategy = configs.class_strategy
        self.use_pretrained_emb = configs.use_pretrained_emb
        self.joint_train = configs.joint_train
        self.supervised_joint_train = configs.supervised_joint_train
        self.device = 'cuda'
        if configs.use_pretrained_emb:
            self.emb_model = encoder_decoder_small_patch(configs).to(self.device)
            self.emb_model.load_state_dict(torch.load(configs.emb_model_path).module.state_dict())
            if not self.joint_train:
                for param in self.emb_model.parameters():
                    param.requires_grad = False
            self.enc_embedding = None
            self.projector = None
        else:
            self.emb_model = None
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                        configs.dropout)
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.tfm_dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.tfm_activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, flag='train', iter=0):
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        B, L, N = x_enc.shape
        res = [torch.empty(0, device=self.device) for _ in range(4)]
        # **********************************************************************************************************************
        # **********************************************************************************************************************
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        if self.use_pretrained_emb:
            emb = self.emb_model(x_enc.reshape(-1, L), flag='ts2z').reshape(B, N, -1)
        else:
            if self.use_norm:
                # Normalization from Non-stationary Transformer
                means = x_enc.mean(1, keepdim=True).detach()
                x_enc = x_enc - means
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_enc /= stdev
            emb = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        # **********************************************************************************************************************
        # **********************************************************************************************************************
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        emb_out, attns = self.encoder(emb, attn_mask=None)
        # **********************************************************************************************************************
        # **********************************************************************************************************************
        # B N E -> B N S -> B S N
        if self.train_strategy == 'x2y' or flag != 'train':
            if self.use_pretrained_emb:
                output = self.emb_model(emb_out.reshape(B*N, -1), flag='z2ts').reshape(B, -1, N)
            else:
                dec_out = self.projector(emb_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
                if self.use_norm:
                    # De-Normalization from Non-stationary Transformer
                    dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                    output = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                else:
                    output = dec_out
            res[0] = output[:, -self.pred_len:, :]  # [B, L, D]
        elif self.train_strategy == 'z2z':
            res[3] = emb_out
        else:
            raise ValueError('train_strategy not recognized')
        # **********************************************************************************************************************
        # **********************************************************************************************************************
        if self.joint_train and self.supervised_joint_train and flag == 'train':
            if iter % 500 == 0:
                x_reconstructed, loss_vae, _, _, _ = self.emb_model(x_enc.reshape(B*N, -1))
                x_reconstructed = x_reconstructed.reshape(B, -1, N)
                res[1] = x_reconstructed
                res[2] = loss_vae
        return tuple(res)

    def ts2z(self, y, use_var=True):
        B, L, N = y.shape
        y = y.reshape(B*N, L)
        return self.emb_model.ts2z(y, use_var=use_var).detach().reshape(B, N, self.d_model)