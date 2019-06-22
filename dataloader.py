from __future__ import print_function, division
import os
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config


def normalize_data(df):
    mean = df.mean()
    std = df.std()
    df = (df - mean) / std
    return df

class EGG_Dataset(Dataset):
    """Face Landmarks dataset."""
    label_set = set()
    mean_list = []
    std_list = []
    Negtive_num = 0
    def __init__(self, path):
        super(EGG_Dataset, self).__init__()
        # load the csv here
        # we will also do the segementation here
        self.eeg_df = pd.read_csv(path)
        self.eeg_df = self.eeg_df[0:int(len(self.eeg_df))]
        self.eeg_df.iloc[:,0:14] = normalize_data(self.eeg_df.iloc[:,0:14])
        self.all_list = []
        self.all_label = []
        last_class = 1
        last_index = 0
        for i in range(len(self.eeg_df)):
            if (self.eeg_df.loc[i,"Class"] != last_class) or (i == (len(self.eeg_df) - 1)): # when the label change, do the split
                tmp_array = self.eeg_df.iloc[last_index:i,0:14].values
                last_index = i
                # slice the previous array
                for j in range((len(tmp_array) -2) // 3): # how many segments
                    segment = tmp_array[3*j:3*j + 5,:]
                    self.all_list.append(segment)
                    self.all_label.append(last_class)
                last_class = self.eeg_df.loc[i,"Class"]  # update teh last_classs
                if last_class not in self.label_set:
                    self.label_set.add(last_class)
        #train_set
        zipped = zip(self.all_list,self.all_label)
        zipped_list = list(zipped)
        random.shuffle(zipped_list)
        self.all_list, self.all_label = zip(*zipped_list)
        train_idx = int(len(self.all_list)*config.TRAIN_RATIO)
        test_idx = int(len(self.all_list)*(config.TRAIN_RATIO + config.TEST_RATIO))
        self.train_list = list(self.all_list[0:train_idx])
        self.train_label = list(self.all_label[0:train_idx])
        self.test_list = list(self.all_list[train_idx:test_idx])
        self.test_label = list(self.all_label[train_idx:test_idx])
        a = 1
    def __len__(self):
        # get the length of the data set
        return len(self.train_list)
    def __getitem__(self, idx):
        # get one sample according to idx
        sample = {}
        sample["seq"] = self.train_list[idx]
        sample["label"] = self.train_label[idx]
        return sample

    def generate_target_sample(self, target_label):
        Flag = True
        while Flag:
            rand_idx = np.random.randint(0, len(self.train_label)-1)
            if self.train_label[rand_idx] == target_label:
                return self.train_list[rand_idx]

    def generate_negative_sample(self,sample_number, target_label):
        tmp_list = set()
        for item in self.label_set:
            if item != target_label:
                tmp_list.add(item)
        sampled_num = 0
        output_list = []
        while sampled_num < sample_number:
            rand_idx = np.random.randint(0, len(self.train_label) - 1)
            if self.train_label[rand_idx] in tmp_list:
                output_list.append(self.train_list[rand_idx])
                sampled_num += 1
        return output_list

    def Num_for_each_class(self):
        label_count_list = {}
        for item in self.label_set:
            label_count_list[str(item)] = self.train_label.count(item)
        return label_count_list

    def get_train_data(self):
        return self.train_list, self.train_label
    def get_test_data(self):
        return self.test_list, self.test_label
if __name__ == "__main__":
    dataset = EGG_Dataset("data/eeg-eye-state_csv.csv")
    test = dataset.Num_for_each_class()
    print(dataset.__len__())
    a = 1