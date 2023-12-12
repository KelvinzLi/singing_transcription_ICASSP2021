# For Multi-Input-Multi-Output Models

from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import concurrent.futures
import pickle
import time
import json

class MimoAudioDataset(Dataset):
    # around 43 samples persecond for the current setting
    def __init__(self, data_dir, window_size=128, stride=32, label_smoother=None, random_shift=True, log_and_normalize=False):
        print(window_size)
        print(stride)
        
        self.data_dir = data_dir
        self.window_size = window_size
        self.random_shift = random_shift
        self.log_and_normalize = log_and_normalize
        
        counter = 0
        
        self.data_instances = []
        self.idx_mapper = []
        for ii, file_name in enumerate(os.listdir(self.data_dir)):
            data = np.load(os.path.join(self.data_dir, file_name))
            cqt_data = torch.tensor(data['x'], dtype=torch.float)
            answer_data = data['y']
            self.data_instances.append((cqt_data, answer_data))
            
            _mapper = []
            if cqt_data.shape[0] < window_size:
                _mapper.append((ii, 0))
            else:
                for jj in range((cqt_data.shape[0] - window_size) // stride + 1):
                    _mapper.append((ii, jj * stride))
            
            self.idx_mapper.extend(_mapper)
            counter += cqt_data.shape[0]
            
        print(len(self.idx_mapper))
        print(counter)
        
        self.label_smoother = label_smoother
        
    def __getitem__(self, idx):
        data_idx, frame_idx = self.idx_mapper[idx]
    
        cqt_data, answer_data = self.data_instances[data_idx]

        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]

        my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)
        my_gt_padding = np.array([0, 1, 4, 12])
        
        if self.random_shift:
            frame_idx = min(frame_idx + random.randint(0, self.window_size - 1), cqt_data.shape[0]-self.window_size//2)
        
        gt = answer_data[frame_idx: frame_idx+self.window_size]
        cqt_feature = cqt_data[frame_idx: frame_idx+self.window_size]
        
        if self.log_and_normalize:
            cqt_feature = torch.clamp(cqt_feature, 1e-6, None)
            cqt_feature = torch.log10(cqt_feature)
            cqt_feature = (cqt_feature - torch.mean(cqt_feature)) / torch.clamp(torch.std(cqt_feature), 1e-5, None)
        
        if len(gt) < self.window_size:
            gt = np.concatenate([gt, [my_gt_padding] * (self.window_size - len(gt))], 0)
            cqt_feature = torch.cat([cqt_feature, torch.stack([my_padding] * (self.window_size - len(cqt_feature)), dim=0)], dim=0)
            
        cqt_feature = torch.permute(cqt_feature, (1, 0, 2))
        
        gt = gt.astype(np.float32)
        if self.label_smoother is not None:
            gt[:, 0] = self.label_smoother(gt[:, 0])
        
        gt = np.transpose(gt)

        return (cqt_feature, gt)

    def __len__(self):
        return len(self.idx_mapper)