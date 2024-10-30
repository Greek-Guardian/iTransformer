import os, time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_provider.data_factory import data_provider
from matplotlib import pyplot as plt

def tfm_train(args, data_loader, dir_path, model, prof=None):
    '''训练'''
    device = 'cuda'

    model.train()
    criterion_mse = nn.MSELoss()
    # HuberLoss
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    with SummaryWriter(log_dir=dir_path) as writer:
        for epoch_count in tqdm(range(250)):
            for ii, (x, y, x_mark, y_mark) in (enumerate(data_loader)):
                B, L, C = x.shape
                x = x.to(device).float()
                y = y.to(device)[:, -args.pred_len:, :].float()
                x_mark = x_mark.to(device).float()
                # x = torch.reshape(x.to(device), shape=[B*C, L]).float()
                # y = torch.reshape(y.to(device)[:, -args.pred_len:, :], shape=[B*C, L]).float()
                optimizer.zero_grad()
                y_pred, x_reconstructed, loss_vae, z_y_apostrophe = model(x, x_mark, y, y_mark, iter=ii)
                if args.train_strategy == 'x2y':
                    smoothL1_loss = criterion(y, y_pred)
                    mse_loss = criterion_mse(y, y_pred)
                    mae_loss = torch.mean(torch.abs(y - y_pred))
                    loss = (mse_loss if args.loss == 'mse' else smoothL1_loss)
                    writer.add_scalar('Loss/loss',            loss.mean().item(),       epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Loss/SmoothL1Loss',    smoothL1_loss.item(),     epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mse',          mse_loss.item(),          epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Metrics/mae',          mae_loss.item(),          epoch_count * len(data_loader) + ii)
                elif args.train_strategy == 'z2z':
                    z_y = model.module.ts2z(y, use_var=True)
                    loss = criterion_mse(z_y, z_y_apostrophe)
                    writer.add_scalar('Loss/loss', loss.mean().item(), epoch_count * len(data_loader) + ii)
                    if ii % 100 == 0:
                        y_pred_eval, _, _, _ = model(x, x_mark, y, y_mark, flag='eval')
                        mse_metric = criterion_mse(y, y_pred_eval)
                        mae_metric = torch.mean(torch.abs(y - y_pred_eval))
                        writer.add_scalar('Metrics/mse',          mse_metric.item(),          epoch_count * len(data_loader) + ii)
                        writer.add_scalar('Metrics/mae',          mae_metric.item(),          epoch_count * len(data_loader) + ii)
                else:
                    raise ValueError('train_strategy not recognized')
                if args.joint_train and args.supervised_joint_train and (x_reconstructed.numel() != 0):
                    supervised_loss_mse = criterion_mse(x, x_reconstructed)
                    loss_vae = loss_vae * args.kld_loss_weight
                    loss = loss + supervised_loss_mse.mean() + loss_vae.mean()
                    writer.add_scalar('Loss/supervised_loss_mse', supervised_loss_mse.mean().item(), epoch_count * len(data_loader) + ii)
                    writer.add_scalar('Loss/supervised_loss_vae', loss_vae.mean().item(), epoch_count * len(data_loader) + ii)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if prof is not None:
                    prof.step()

def save_plot(fig, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, filename))

def tfm_eval(args, model, save_path, device='cuda'):
    '''画出结果'''
    model.eval()

    for flag in ['train', 'val', 'test']:
        data_set, data_loader = data_provider(args, flag=flag)

        for ii, (x, y, x_mark, y_mark) in enumerate(data_loader):
            B, L, C = x.shape
            x = x.to(device).float()
            y = y.to(device)[:, -args.pred_len:, :].float()
            x_mark = x_mark.to(device).float()
            # x = torch.reshape(x.to(device), shape=[B*C, L]).float()
            # y = torch.reshape(y.to(device)[:, -args.pred_len:, :], shape=[B*C, L]).float()
            y_pred, x_reconstructed, loss_vae, z_y_apostrophe  = model(x_enc=x, x_mark_enc=x_mark, x_dec=y, x_mark_dec=y_mark, flag='eval')
            criterion = nn.MSELoss()
            mse = criterion(y, y_pred)
            mae = torch.mean(torch.abs(y - y_pred))
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            # 在同一张图上，按照列添加子图，画出前五个原始数据和重构数据
            fig, axs = plt.subplots(5, 1, figsize=(10, 10))
            try:
                for i in range(5):
                    axs[i].plot(y[0, :, i], label='Original')
                    axs[i].plot(y_pred[0, :, i], label='Reconstructed')
                    axs[i].legend()
            except:
                pass
            plt.tight_layout()
            plt.show()
            save_plot(fig, save_path, flag+"_mse{:.3f}_".format(mse.item())+'mae{:.3f}_'.format(mae.item())+'.png')
            break