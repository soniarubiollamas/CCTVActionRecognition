import torch.utils.data as data
from utils import process_feat
import numpy as np
import os


class UCFDataset(data.Dataset):
    def __init__(self, cfg, transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = 'Normal'
        self.abnormal_dict = {'Normal':0,'Abuse':1, 'Arrest':2, 'Arson':3, 'Assault':4,
                              'Burglary':5, 'Explosion':6, 'Fighting':7,'RoadAccidents':8,
                              'Robbery':9, 'Shooting':10, 'Shoplifting':11, 'Stealing':12, 'Vandalism':13}
        self.t_features = np.array(np.load(cfg.token_feat))
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        video_name = self.list[index].strip('\n').split('/')[-1][:-4]
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        video_idx = self.list[index].strip('\n').split('/')[-1].split('_')[0]
        if self.normal_flag in self.list[index]:
            video_ano = video_idx
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = video_idx[:-3]
            ano_idx = self.abnormal_dict[video_ano]
            label = 1.0

        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        fg_feat = np.array(self.t_features[ano_idx, :], dtype=np.float16)
        bg_feat = np.array(self.t_features[0, :], dtype=np.float16)
        fg_feat = fg_feat.reshape(1, 512)
        bg_feat = bg_feat.reshape(1, 512)
        t_feat = np.concatenate((bg_feat, fg_feat), axis=0)
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
            t_feat = self.tranform(t_feat)

        if self.test_mode:
            filename = self.list[index].strip('\n')
            return v_feat, filename #, ano_idx #, video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, t_feat, label, ano_idx

    def __len__(self):
        return len(self.list)
