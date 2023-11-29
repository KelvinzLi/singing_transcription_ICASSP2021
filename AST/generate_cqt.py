from data_utils.audio_dataset import preprocess, get_cqt

import argparse
from pathlib import Path

from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import pickle
import time
import json

parser = argparse.ArgumentParser("")
parser.add_argument('data_dir')
parser.add_argument('gt_path')
parser.add_argument('output_dir')
parser.add_argument('-s', '--split_txt', default=None)

def get_feature(y):
    y = librosa.util.normalize(y)
    cqt_feature = get_cqt(y, 1.0)
    return np.expand_dims(cqt_feature, axis = 1)

def generate_cqt(gt_path, data_dir, output_dir, split_txt=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    with open(gt_path) as json_data:   
        gt = json.load(json_data)

    if split_txt is not None:
        with open(split_txt, 'r') as txt:
            split = txt.read()
        split = split.split("\n")
    else:
        print('\n')
        print('Split information not given! Using all the data!')
        print('\n')
        split = [file_id.split('.')[0] for file_id in os.listdir(data_dir)]

    total_count = len(split)

    temp_cqt = {}
    future = {}
    print ("computing CQT......")

    frame_size = 1024.0 / 44100.0

    for the_dir in tqdm(split):
        wav_path = os.path.join(data_dir, the_dir + '.wav')

        y, sr = librosa.core.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
            
        cqt_data = get_feature(y)       

        gt_data = gt[the_dir]
        answer_data = preprocess(gt_data, cqt_data.shape[0])
        
        np.savez(os.path.join(output_dir, the_dir + 'npz'), x=cqt_data, y=answer_data)
        

if __name__ == "__main__":
    args = parser.parse_args()

    generate_cqt(args.gt_path, args.data_dir, args.output_dir, args.split_txt)