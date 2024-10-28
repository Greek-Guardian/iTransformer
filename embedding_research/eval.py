import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from matplotlib import pyplot as plt
import os

def save_plot(fig, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, filename))

def eval(args, enc_dec_small_patch, save_path, device='cuda'):
    '''画出结果'''
    enc_dec_small_patch.eval()

    for flag in ['train', 'val', 'test']:
        data_set, data_loader = data_provider(args, flag=flag)

        for ii, (x, y, x_mark, y_mark) in enumerate(data_loader):
            B, L, C = x.shape
            x = torch.reshape(x, shape=[B*C, L]).to(device).float()
            y = torch.reshape(y[:, -args.pred_len:, :], shape=[B*C, L]).to(device).float()
            x_recover, _, _, _, _ = enc_dec_small_patch(x)
            criterion = nn.MSELoss()
            mse = criterion(x_recover, x)
            x = x.cpu().detach().numpy()
            x_recover = x_recover.cpu().detach().numpy()
            # 在同一张图上，按照列添加子图，画出前五个原始数据和重构数据
            fig, axs = plt.subplots(5, 1, figsize=(10, 10))
            for i in range(5):
                axs[i].plot(x[i], label='Original')
                axs[i].plot(x_recover[i], label='Reconstructed')
                axs[i].legend()
            plt.tight_layout()
            plt.show()
            save_plot(fig, save_path, "{:.3f}".format(mse.item())+'_'+flag+'_reconstructed_data.png')
            break