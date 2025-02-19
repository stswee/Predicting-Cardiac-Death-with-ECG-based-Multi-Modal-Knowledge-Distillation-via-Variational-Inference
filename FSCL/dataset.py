import os
import random
import numpy as np

import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import wfdb
import pandas as pd

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0

class CelebaLoader(Dataset):
    def __init__(self,split,ta,ta2,sa,sa2,data_folder,transform):
        self.data_folder=data_folder
        if split==0:
            self.img_list=os.listdir(self.data_folder+'train')
        elif split==1:
            self.img_list=os.listdir(self.data_folder+'val')
        else :
            self.img_list=os.listdir(self.data_folder+'test')

        self.split=split
        self.img_list.sort()
        self.transform=transform
        self.att=[]
        

        with open(self.data_folder+'list_attr_celeba.csv','r') as f:
            reader=csv.reader(f)
            att_list=list(reader)
        att_list=att_list[1:]

        with open(self.data_folder+'list_eval_partition.csv','r') as f:
            reader=csv.reader(f)
            eval_list=list(reader)
    
        for i,eval_inst in enumerate(eval_list):
            if eval_inst[1]==str(self.split):
                if att_list[i][0]==eval_inst[0]:
                    self.att.append(att_list[i])
                else:
                    pass

        
        self.att=np.array(self.att)
        self.att=(self.att=='1').astype(int)
        self.ta=ta
        self.ta2=ta2
        self.sa=sa
        self.sa2=sa2

    def __getitem__(self, index1):
        
        ta=self.att[index1][int(self.ta)]
        sa=self.att[index1][int(self.sa)]

        if self.ta2!='None':
            ta2=self.att[index1][int(self.ta2)]
            ta=ta+2*ta2

        if self.sa2!='None':
            sa2=self.att[index1][int(self.sa2)]
            sa=sa+2*sa2

        
        index2=random.choice(range(len(self.img_list)))
        if self.split==0:
            img1=ecg_signal.open(self.data_folder+'train/'+self.img_list[index1])
            img2=ecg_signal.open(self.data_folder+'train/'+self.img_list[index2])
        elif self.split==1:
            img1=ecg_signal.open(self.data_folder+'val/'+self.img_list[index1])
            img2=ecg_signal.open(self.data_folder+'val/'+self.img_list[index2])
        else:
            img1=ecg_signal.open(self.data_folder+'test/'+self.img_list[index1])
            img2=ecg_signal.open(self.data_folder+'test/'+self.img_list[index2])
    
     
        return self.transform(img1),ta,sa


    def __len__(self):
        return len(self.att)


class HolterECGLoader(Dataset):
    def __init__(self, csv_file, ecg_dir, transform=None):
        self.dataset = pd.read_csv(csv_file, sep=";")
        self.ecg_dir = ecg_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ecg_filename = os.path.join(self.ecg_dir, self.dataset.iloc[idx, 0])
        # ecg_signal = wfdb.rdrecord(ecg_filename)
        ecg_signal, _ = wfdb.rdsamp(ecg_filename)

        total_samples = ecg_signal.shape[0]
        target_samples = 14500000 # just take these because every ECG is abit uneven
        if total_samples >= target_samples:
            start_idx = (total_samples - target_samples) // 2 
            end_idx = start_idx + target_samples
            ecg_signal = ecg_signal[start_idx:end_idx]
        else:
            raise ValueError(f"ECG {ecg_filename} has only {total_samples} samples, which is too short for cropping.")

        # if self.transform:
        #     ecg_signal = self.transform(ecg_signal)
        return ecg_signal