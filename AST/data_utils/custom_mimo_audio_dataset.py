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
    def __init__(self, data_dir, window_size=128, onset_decay=0.875, stride=32, random_shift=True, kernel_size=1):
        print(window_size)
        print(onset_decay)
        print(stride)
        
        self.data_dir = data_dir
        self.window_size = window_size
        self.onset_decay = onset_decay
        self.random_shift = random_shift
        
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
        
        # onset_thres = 0.4
#         if onset_decay == 0:
#             kernel_side_size = 0
#         else:
#             kernel_side_size = int(np.floor(np.log(onset_thres) / np.log(onset_decay)))
#         kernel_size = 1 + 2 * kernel_side_size
        
#         print(kernel_size)
        
#         kernel = np.zeros((kernel_size,))
#         kernel[kernel_side_size] = 1
#         for ii in range(1, kernel_side_size + 1):
#             val = np.power(onset_decay, ii)
#             kernel[kernel_side_size - ii] = val
#             kernel[kernel_side_size + ii] = val
            
        self.kernel = np.ones((kernel_size,))
        print(self.kernel)
        
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
        
        if len(gt) < self.window_size:
            gt = np.concatenate([gt, [my_gt_padding] * (self.window_size - len(gt))], 0)
            cqt_feature = torch.cat([cqt_feature, torch.stack([my_padding] * (self.window_size - len(cqt_feature)), dim=0)], dim=0)
            
        cqt_feature = torch.permute(cqt_feature, (1, 0, 2))
        
        gt = gt.astype(np.float32)
        gt[:, 0] = np.clip(np.convolve(gt[:, 0], self.kernel, 'same'), 0, 1)
        
        gt = np.transpose(gt)

        return (cqt_feature, gt)

    def __len__(self):
        return len(self.idx_mapper)