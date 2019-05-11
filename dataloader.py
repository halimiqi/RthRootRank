from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class EGG_Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, path):
        super(EGG_Dataset, self).__init__()
        # load the csv here
        # we will also do the segementation here
        self.eeg_df = pd.read_csv(path)
        self.all_list = []
        self.all_label = []
        last_class = 1
        last_index = 0
        for i in range(len(self.eeg_df)):
            if self.eeg_df.loc[i,"Class"] != last_class:
                tmp_array = self.eeg_df.iloc[last_index:i,0:14].values
                # slice the previous array
                for j in range((len(tmp_array) -2) // 3): # how many segments
                    segment = tmp_array[3*j:3*j + 5,:]
                    self.all_list.append(segment)
                    self.all_label.append(last_class)
                last_class = self.eeg_df.loc[i,"Class"]
        a = 1
    def __len__(self):
        # get the length of the data set
        return len(self.all_list)
    def __getitem__(self, idx):
        # get one sample according to idx
        sample = {}
        sample["seq"] = self.all_list[idx]
        sample["label"] = self.all_label[idx]
        return sample

    def generate_target_sample(self, target_label):
        Flag = True
        while Flag:
            rand_idx = np.random.randint(0, len(self.all_label)-1)
            if self.all_label[rand_idx] == target_label:
                return self.all_list[rand_idx]


if __name__ == "__main__":
    dataset = EGG_Dataset("data/eeg-eye-state_csv.csv")
