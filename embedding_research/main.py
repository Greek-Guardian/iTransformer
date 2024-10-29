import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
import os, time, json, sys
os.chdir('/home/liangzida/workspace/iTransformer') # 更改工作目录到项目根目录
sys.path.append('/home/liangzida/workspace/iTransformer') # 添加模块路径到 sys.path
from data_provider.data_factory import data_provider
from backbone import encoder_decoder_small_patch
from train import train
from eval import eval

class Args():
    def __init__(self):
        self.data = 'custom'
        self.root_path = './dataset/electricity/'
        self.data_path = 'electricity.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'

        self.batch_size = 32
        self.num_workers = 0
        self.embed = 'timeF'

        self.label_len = 1
        self.seq_len = 96
        self.pred_len = 96

        self.d_model = 512
        self.enc_layers=1
        self.dec_layers=1
        self.dropout=0.5
        self.bidirectional=True
        self.enc_cnn_layer1_dim=32
        self.lstm_num_layers=2
        self.lstm_hidden_size=8
        self.lstm_resnet=True
        self.kld_loss_weight=0.00025

        self.loss = 'mse'
        self.structure = 'VAE' # optionlal: 'Normal', 'VAE'

        self.use_profiler = False

def save(args, enc_dec_small_patch, dir_path):
    torch.save(enc_dec_small_patch, dir_path + '/model.pth')
    with open(dir_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

if __name__ == '__main__':
    args = Args()
    device = 'cuda'
    dir_path = '/home/liangzida/workspace/iTransformer/junk/encdec/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'\
                    + 'seqlen' + str(args.seq_len) + 'd_model' + str(args.d_model) + 'enc_layers' + str(args.enc_layers)\
                    + 'dec_layers' + str(args.dec_layers)
    os.makedirs(dir_path)

    with torch.profiler.profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_path + '/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) if args.use_profiler else nullcontext() as prof:
        data_set, data_loader = data_provider(args, flag='train')
        enc_dec_small_patch = encoder_decoder_small_patch(input_dim=args.seq_len, d_model=args.d_model, output_dim=args.seq_len, \
                                                            structure=args.structure,\
                                                            enc_layers=args.enc_layers, dec_layers=args.dec_layers, dropout=args.dropout, enc_cnn_layer1_dim=args.enc_cnn_layer1_dim,\
                                                            bidirectional=args.bidirectional, lstm_num_layers=args.lstm_num_layers, lstm_hidden_size=args.lstm_hidden_size,\
                                                            lstm_resnet=args.lstm_resnet).to(device)
        enc_dec_small_patch = nn.parallel.DataParallel(enc_dec_small_patch, device_ids=[0, 1])
        try:
            train(args, data_loader, dir_path, enc_dec_small_patch, prof)
            eval(args, enc_dec_small_patch, dir_path)
            save(args, enc_dec_small_patch, dir_path)
            print("Training ends. Model saved.")
        except KeyboardInterrupt:
            eval(args, enc_dec_small_patch, dir_path)
            save(args, enc_dec_small_patch, dir_path)
            print("Program interrupted. Model saved.")
        except Exception as e:
            eval(args, enc_dec_small_patch, dir_path)
            save(args, enc_dec_small_patch, dir_path)
            print("Program interrupted. Model saved.")
            print(f"An error occurred: {e}")