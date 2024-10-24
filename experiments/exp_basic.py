import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer
import time


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # setting record of experiments
        model_structure_dict = {
            0: 'all',
            1: 'noemb',
            2: 'noenc',
            3: 'noproj',
            4: 'noembnoenc',
            5: 'noembnoproj',
            6: 'noencnoproj',
            # 0: 'all',
            # 1: 'encoder',
            # 2: 'decoder',
            # 3: 'attention',
            # 4: 'ffn',
            # 5: 'all_but_ffn',
            # 6: 'all_but_attention',
            # 7: 'all_but_pos',
            # 8: 'all_but_pos_ffn',
            # 9: 'all_but_pos_attention',
            # 10: 'all_but_pos_ffn_attention',
        }
        self.setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            model_structure_dict[args.model_structure])
        self.start_time = time.time()
        self.save_dir = os.path.join('./outputs/', self.setting, time.strftime("%m-%d-%H-%M", time.localtime(self.start_time)))
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
