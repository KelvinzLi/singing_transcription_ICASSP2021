import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
from importlib import import_module

import sys
import os

from net import EffNetb0
import math
from data_utils import AudioDataset

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)

def _pooling_prep(x, kernel_size):
    # assume x (batch_size, time)
    assert kernel_size % 2 == 1
    
    kernel_side_size = kernel_size // 2
    batch_size, time_length = x.shape
    
    x = np.concatenate([np.zeros((batch_size, kernel_side_size)),
                       x,
                       np.zeros((batch_size, kernel_side_size))], axis=1)
    x = np.expand_dims(x, axis=-1)
    
    x_slices = [x[:, ii: ii + time_length] for ii in range(kernel_size)]
    return np.concatenate(x_slices, axis=-1)

def maxpool1d(x, kernel_size):
    # assume x (batch_size, time)
    return np.max(_pooling_prep(x, kernel_size), axis=-1)

def sumpool1d(x, kernel_size):
    # assume x (batch_size, time)
    return np.sum(_pooling_prep(x, kernel_size), axis=-1)

def tp_tn_analysis(y, gt, kernel_size):
    tn_mask = 1 - 1 * (sumpool1d(gt, kernel_size) >= 1)
    
    tp_ratio = np.sum(gt * maxpool1d(y, kernel_size)) / np.sum(gt)
    tn_ratio = np.sum(tn_mask * (1 - y)) / np.sum(tn_mask)
    
    return tp_ratio, tn_ratio

class Predictor:
    def __init__(self, device= "cuda:0", model_path=None, model_import_path='net.EffNetb0', model_kwargs={}):
        """
        Params:
        model_path: Optional pretrained model file
        """
        
        # Initialize model
        self.device = device
        
        model_builder = getattr(import_module('.'.join(model_import_path.split('.')[:-1])), model_import_path.split('.')[-1])

        self.model = model_builder(**model_kwargs).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            print('Model read from {}.'.format(model_path))

        print('Predictor initialized with {}.'.format(model_import_path))


    def fit(
        self, 
        train_dataset_path, 
        valid_dataset_path, model_dir, 
        dataset_import_path='data_utils.AudioDataset', 
        dataset_kwargs={}, 
        label_smoother_import_path=None, 
        label_smoother_kwargs={},
        weight_decay=0, 
        onset_pos_weight=15.0,
        offset_pos_weight=15.0,
        loss_weights = [1.2, 1.2, 0.8, 0.8],
        **training_args
    ):
        """
        train_dataset_path: The path to the training dataset.pkl
        valid_dataset_path: The path to the validation dataset.pkl
        model_dir: The directory to save models for each epoch
        training_args:
          - batch_size
          - valid_batch_size
          - epoch
          - lr
          - save_every_epoch
        """
        
        # Set paths
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Set training params
        self.batch_size = training_args['batch_size']
        self.valid_batch_size = training_args['valid_batch_size']
        self.epoch = training_args['epoch']
        self.lr = training_args['lr']
        self.save_every_epoch = training_args['save_every_epoch']
        
        self.train_smoother = None
        if label_smoother_import_path is not None:
            smoother_builder = getattr(import_module('.'.join(label_smoother_import_path.split('.')[:-1])), label_smoother_import_path.split('.')[-1])
            self.train_smoother = smoother_builder(**label_smoother_kwargs)
        
        
        dataset_builder = getattr(import_module('.'.join(dataset_import_path.split('.')[:-1])), dataset_import_path.split('.')[-1])
        self.training_dataset = dataset_builder(self.train_dataset_path, label_smoother=self.train_smoother, **dataset_kwargs)
        self.validation_dataset = dataset_builder(self.valid_dataset_path, label_smoother=None, **dataset_kwargs)

        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.valid_batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        start_epoch = 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.epoch * len(self.train_loader) + 1)
        
        if 'checkpoint.pth' in os.listdir(model_dir):
            print('training model from checkpoint ......')
            
            checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
            checkpoint = torch.load(checkpoint_path)

            start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['opt'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(onset_pos_weight),], device=self.device))
        self.offset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(offset_pos_weight),], device=self.device))

        self.octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)

        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))

        start_time = time.time()
        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        print (self.iters_per_epoch)

        print_keys = ('loss', 'tp', 'tn')
        
        for epoch in range(start_epoch, self.epoch + 1):
            
            train_history = []
            
            self.model.train()

            total_training_loss = 0
            total_split_loss = np.zeros(4)            

            with tqdm(self.train_loader, unit=" batch") as pbar:
                pbar.set_description('epoch_{}'.format(epoch))
                for batch_idx, batch in enumerate(pbar):
                    # Parse batch data

                    input_tensor = batch[0].to(self.device)
                    onset_prob = batch[1][:, 0].float().to(self.device)
                    offset_prob = batch[1][:, 1].float().to(self.device)
                    pitch_octave = batch[1][:, 2].long().to(self.device)
                    pitch_class = batch[1][:, 3].long().to(self.device)

                    loss = 0                
                    self.optimizer.zero_grad()

                    onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)
                    

                    split_train_loss0 = self.onset_criterion(onset_logits, onset_prob)
                    split_train_loss1 = self.offset_criterion(offset_logits, offset_prob)
                    split_train_loss2 = self.octave_criterion(pitch_octave_logits, pitch_octave)
                    split_train_loss3 = self.pitch_criterion(pitch_class_logits, pitch_class)
                    
                    if split_train_loss0 < 0:
                        print(0)
                    if split_train_loss1 < 0:
                        print(1)
                    if split_train_loss2 < 0:
                        print(2)
                    if split_train_loss3 < 0:
                        print(3)

                    # print(onset_prob.shape)
                    # print(onset_logits.size())
                    # print(onset_prob[0, :10])
                    # print(torch.sigmoid(onset_logits)[0, :10].detach().cpu().numpy())

                    total_split_loss[0] = total_split_loss[0] + split_train_loss0.item() 
                    total_split_loss[1] = total_split_loss[1] + split_train_loss1.item()
                    total_split_loss[2] = total_split_loss[2] + split_train_loss2.item()
                    total_split_loss[3] = total_split_loss[3] + split_train_loss3.item()

                    loss = loss_weights[0]*split_train_loss0 + loss_weights[1]*split_train_loss1 + loss_weights[2]*split_train_loss2 + loss_weights[3]*split_train_loss3
                    loss.backward()
                    self.optimizer.step()
                    
                    onset_prob_np = 1 * (batch[1][:, 0].float().numpy() > 0.2)
                    onset_pred_np = 1 * (torch.sigmoid(onset_logits).detach().cpu().numpy() > 0.5)
                    
                    true_pos_onset, true_neg_onset = tp_tn_analysis(onset_pred_np, onset_prob_np, 11)
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.scheduler is not None:
                        self.scheduler.step()
                    total_training_loss += loss.item()
                    
                    logs = {'loss': loss.item(), 'lr': current_lr, 'tp': true_pos_onset, 'tn': true_neg_onset}
                    train_history.append(logs)

                    pbar.set_postfix({k: logs[k] for k in print_keys})

                    # if batch_idx % 5000 == 0 and batch_idx != 0:
                    #     print (epoch, batch_idx, "time:", time.time()-start_time, "loss:", total_training_loss / (batch_idx+1))


            if epoch % self.save_every_epoch == 0:
                # Perform validation
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    split_val_loss = np.zeros(6)
                    
                    total_valid_tp_onset = 0
                    total_valid_tn_onset = 0
                    for batch_idx, batch in enumerate(self.valid_loader):

                        input_tensor = batch[0].to(self.device)

                        onset_prob = batch[1][:, 0].float().to(self.device)
                        offset_prob = batch[1][:, 1].float().to(self.device)
                        pitch_octave = batch[1][:, 2].long().to(self.device)
                        pitch_class = batch[1][:, 3].long().to(self.device)

                        onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

                        split_val_loss0 = self.onset_criterion(onset_logits, onset_prob)
                        split_val_loss1 = self.offset_criterion(offset_logits, offset_prob)

                        split_val_loss2 = self.octave_criterion(pitch_octave_logits, pitch_octave)
                        split_val_loss3 = self.pitch_criterion(pitch_class_logits, pitch_class)

                        split_val_loss[0] = split_val_loss[0] + split_val_loss0.item()
                        split_val_loss[1] = split_val_loss[1] + split_val_loss1.item()
                        split_val_loss[2] = split_val_loss[2] + split_val_loss2.item()  
                        split_val_loss[3] = split_val_loss[3] + split_val_loss3.item()

                        
                        # Calculate loss
                        loss = loss_weights[0]*split_val_loss0 + loss_weights[1]*split_val_loss1 + loss_weights[2]*split_val_loss2 + loss_weights[3]*split_val_loss3
                        total_valid_loss += loss.item()
                        
                        onset_prob_np = 1 * (batch[1][:, 0].float().numpy() > 0.2)
                        onset_pred_np = 1 * (torch.sigmoid(onset_logits).detach().cpu().numpy() > 0.5)

                        true_pos_onset, true_neg_onset = tp_tn_analysis(onset_pred_np, onset_prob_np, 11)
                        
                        total_valid_tp_onset += true_pos_onset
                        total_valid_tn_onset += true_neg_onset


                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_dir) / (training_args['save_prefix']+'_{}'.format(epoch))
                torch.save(save_dict, target_model_path)
                
                # Save Checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'opt': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(self.model_dir, 'checkpoint.pth'))
                
                # Save Training History
                df = pd.DataFrame(train_history)
                df.to_csv(os.path.join(self.model_dir, 'epoch_{}.csv'.format(epoch)))
                
                val_log = pd.DataFrame([{'epoch': epoch, 'iters': len(self.train_loader) * epoch, 
                                         'loss': total_valid_loss / len(self.valid_loader), 
                                         'tp': total_valid_tp_onset / len(self.valid_loader), 
                                         'tn': total_valid_tn_onset / len(self.valid_loader)}])
                
                val_history = None
                val_file_name = 'validation_history.csv'
                if val_file_name in os.listdir(self.model_dir):
                    val_history = pd.read_csv(os.path.join(self.model_dir, val_file_name))
                val_history = val_log if val_history is None else pd.concat([val_history, val_log])
                val_history.to_csv(os.path.join(self.model_dir, val_file_name))
                

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        total_training_loss / len(self.train_loader),
                        total_valid_loss / len(self.valid_loader),
                        time.time()-start_time))

                print('split train loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch class {:.4f}'.format(
                        total_split_loss[0]/len(self.train_loader),
                        total_split_loss[1]/len(self.train_loader),
                        total_split_loss[2]/len(self.train_loader),
                        total_split_loss[3]/len(self.train_loader)
                    )
                )
                print('split val loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch class {:.4f}'.format(
                        split_val_loss[0]/len(self.valid_loader),
                        split_val_loss[1]/len(self.valid_loader),
                        split_val_loss[2]/len(self.valid_loader),
                        split_val_loss[3]/len(self.valid_loader)
                    )
                )
                print('onset: true positive ratio {:.4f} true false negative {:.4f}'.format(
                        total_valid_tp_onset / len(self.valid_loader),
                        total_valid_tn_onset / len(self.valid_loader),
                    )
                )
        
        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def _parse_frame_info(self, frame_info, onset_thres, offset_thres, return_details=False):
        """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""

        result = []
        current_onset = None
        pitch_counter = []

        last_onset = 0.0
        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

        local_max_size = 3
        current_time = 0.0

        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time
                    last_onset = info[0] - onset_thres

                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, np.mean(pitch_counter, axis=0)])

                    current_onset = current_time
                    last_onset = info[0] - onset_thres
                    pitch_counter = []

            elif info[1] >= offset_thres:  # If is offset
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, np.mean(pitch_counter, axis=0)])
                    current_onset = None

                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                pitch_octave_idx = np.argmax(info[2])
                pitch_class_idx = np.argmax(info[3])
                
                if pitch_octave_idx != 4 and pitch_octave_idx != 12:
                # if final_pitch != 60:
                    pitch_octave_details = info[2][:-1] / np.sum(info[2][:-1])
                    pitch_class_details = info[3][:-1] / np.sum(info[3][:-1])
                    pitch_counter.append(np.concatenate([pitch_octave_details, pitch_class_details], axis=0))

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, current_time, np.mean(pitch_counter, axis=0)])
            current_onset = None
            
        if not return_details:
            for ii, info in enumerate(result):
                info[2] = np.argmax(info[2][:-12]) * 12 + np.argmax(info[2][-12:]) + 36
                result[ii] = info

        return result

    def predict(self, test_dataset, results={}, onset_thres=0.1, offset_thres=0.5, mimo=False, return_details=False):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 500
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            raw_data = {}
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                song_ids = batch[1]

                result_tuple = self.model(input_tensor)
                onset_logits = result_tuple[0]
                offset_logits = result_tuple[1]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                pitch_octave_logits, pitch_class_logits = torch.nn.functional.softmax(pitch_octave_logits.cpu(), dim=1), torch.nn.functional.softmax(pitch_class_logits.cpu(), dim=1)
                # print (pitch_octave_logits)


                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    if not mimo:
                        frame_info = (onset_probs[bid], offset_probs[bid], pitch_octave_logits[bid], pitch_class_logits[bid])
                    else:
                        frame_info = (onset_probs[bid].numpy(), offset_probs[bid].numpy(), pitch_octave_logits[bid].numpy().T, pitch_class_logits[bid].numpy().T)

                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)
                    
            if mimo:
                for song_id, frame_infos in song_frames_table.items():
                    onset_infos = test_dataset.stitch([f[0] for f in frame_infos])
                    offset_infos = test_dataset.stitch([f[1] for f in frame_infos])
                    pitch_octave_infos = test_dataset.stitch([f[2] for f in frame_infos])
                    pitch_class_infos = test_dataset.stitch([f[3] for f in frame_infos])
                    
                    stitched_frame_infos = list(zip(onset_infos, 
                                                    offset_infos, 
                                                    pitch_octave_infos, 
                                                    pitch_class_infos
                                                   )
                                               )
                    song_frames_table[song_id] = stitched_frame_infos

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = self._parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres, return_details=return_details)
        return results