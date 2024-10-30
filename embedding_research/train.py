import os, time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(args, data_loader, dir_path, enc_dec_small_patch, prof=None):
    '''训练'''
    device = 'cuda'

    enc_dec_small_patch.train()
    criterion_mse = nn.MSELoss()
    # HuberLoss
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(enc_dec_small_patch.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    with SummaryWriter(log_dir=dir_path) as writer:
        for epoch_count in tqdm(range(250)):
            for ii, (x, y, x_mark, y_mark) in (enumerate(data_loader)):
                B, L, C = x.shape
                x = torch.reshape(x.to(device), shape=[B*C, L]).float()
                y = torch.reshape(y.to(device)[:, -args.pred_len:, :], shape=[B*C, L]).float()
                optimizer.zero_grad()
                if args.structure == 'Normal':
                    x_recover = enc_dec_small_patch(x)
                    target = x if args.task == 'reconstruct' else y
                    smoothL1_loss = criterion(x_recover, target)
                    mse_loss = criterion_mse(x_recover, target)
                    mae_loss = torch.mean(torch.abs(x_recover - target))
                    loss = (mse_loss if args.loss == 'mse' else smoothL1_loss)
                    writer.add_scalar('Loss/loss',            loss.mean().item(),       epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Loss/SmoothL1Loss',    smoothL1_loss.item(),     epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mse',          mse_loss.item(),          epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mae',          mae_loss.item(),          epoch_count * len(data_loader) + ii)
                elif args.structure == 'VAE':
                    x_recover, loss_vae, mean, var, z = enc_dec_small_patch(x)
                    loss_vae = loss_vae * args.kld_loss_weight
                    target = x if args.task == 'reconstruct' else y
                    smoothL1_loss = criterion(x_recover, target)
                    mse_loss = criterion_mse(x_recover, target)
                    mae_loss = torch.mean(torch.abs(x_recover - target))
                    loss = loss_vae.mean() + (mse_loss if args.loss == 'mse' else smoothL1_loss)
                    writer.add_scalar('Loss/loss',            loss.mean().item(),       epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Loss/SmoothL1Loss',    smoothL1_loss.item(),     epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Loss/VAEloss',         loss_vae.mean().item(),   epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mse',          mse_loss.item(),          epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mae',          mae_loss.item(),          epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Latent/mean',         mean.abs().mean().item(), epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Latent/var',          var.mean().item(),        epoch_count * len(data_loader) + ii)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if prof is not None:
                    prof.step()