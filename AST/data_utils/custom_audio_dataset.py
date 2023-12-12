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

class AudioDataset(Dataset):
    def __init__(self, data_dir, window_size=11, onset_decay=0.875, label_smoother=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.onset_decay = onset_decay
        
        self.data_instances = []
        self.idx_mapper = []
        for ii, file_name in enumerate(os.listdir(self.data_dir)):
            data = np.load(os.path.join(self.data_dir, file_name))
            cqt_data = torch.tensor(data['x'], dtype=torch.float)
            answer_data = data['y']
            self.data_instances.append((cqt_data, answer_data))
            self.idx_mapper.extend([(ii, jj) for jj in range(cqt_data.shape[0])])
        
    def __getitem__(self, idx):
        data_idx, frame_idx = self.idx_mapper[idx]
    
        cqt_data, answer_data = self.data_instances[data_idx]

        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]

        my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)
        
        # frame_idx = random.randint(0, frame_num - 1)
        
        gt = answer_data[frame_idx]

        cqt_feature = []
        for frame_window_idx in range(frame_idx-self.window_size//2, frame_idx+self.window_size//2+1):
            # padding with zeros if needed
            if frame_window_idx < 0 or frame_window_idx >= frame_num:
                cqt_feature.append(my_padding.unsqueeze(1))
            else:
                choosed_idx = frame_window_idx
                cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
                
                if answer_data[frame_window_idx][0] == 1:
                    gt[0] = max(gt[0], np.power(self.onset_decay, abs(frame_window_idx - frame_idx)))
                    gt[0] = gt[0] if gt[0] > 0.4 else 0

        cqt_feature = torch.cat(cqt_feature, dim=1)
        return (cqt_feature, gt)

    def __len__(self):
        return len(self.idx_mapper)