# coding:utf-8
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from vis import *
from sklearn.model_selection import train_test_split

class GuiderDataset(Dataset):
    def __init__(self, dir_path, test_size, max_len, min_len=20, target_inter=0.8):
        self.dir = dir_path
        self.max_len = max_len  # use to padding all sequence to a fixed length
        self.min_len = min_len  # use to delete sequence with too many points
        self.target_inter = target_inter
        self.pic_dir = os.path.join(self.dir, "pics/")
        self.seq_dir = os.path.join(self.dir, "sequences/")
        self.pic_name = []
        self.data = []  # use to store all seq
        self.train_data = []
        self.test_data = []
        self.trans = torchvision.transforms.ToTensor()

        for image in os.listdir(self.pic_dir):
            image_name = image.rsplit(".")[0]
            self.pic_name.append(image_name)

        # data preprocess:
        # 1. normalize all coordinate according to the first element(x,y,w,h) in each npy file
        # 2. append the image file name to each coordinate sequence
        # 3. concate all sequence in npy file into one
        for k, image_name in enumerate(self.pic_name):
            data_path = os.path.join(self.seq_dir, image_name + ".npy")
            if not os.path.exists(data_path):
                continue
            raw_data = np.load(
                data_path, allow_pickle=True
            )

            for seq in raw_data:
                # remove zero padding
                seq_x = np.trim_zeros(seq[:,0], trim='b')
                seq_y = np.trim_zeros(seq[:,1], trim='b')

                # for the rare cases where a coordinate is exactly 0.0 and gets trimmed by error
                if len(seq_x) > len(seq_y):
                    seq_y = np.pad(seq_y, pad_width=(0, len(seq_x) - len(seq_y)), constant_values=0.0)
                if len(seq_y) > len(seq_x):
                    seq_x = np.pad(seq_x, pad_width=(0, len(seq_y) - len(seq_x)), constant_values=0.0)

                seq = np.vstack((seq_x, seq_y)).T

                seq = seq.tolist()      # in order to be able to use append                
                if len(seq) > self.min_len:
                    # omit too long
                    continue
                if cal_dis(seq) < 0.2:
                    continue  # delete seq too short
                if intervals_avg(seq) > 1:
                    continue

                seq.append(
                    image_name
                )  # append cooresponding map image name to each sequence

                self.data.append(seq) # seq is a list of list

        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=0)
        print("=" * 50)
        print("Data Preprocess Done!")
        print(
            "Dataset size:{}, train:{}, val:{}".format(
                len(self.data), len(self.train_data), len(self.test_data)
            )
        )
        print("=" * 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = copy.deepcopy(self.data[item])  # a list
        seq_len = len(seq) - 1  # except the last filename element

        trans = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_name = seq[-1]
        seq = seq[:-1]
        image_path = os.path.join(self.pic_dir, image_name + ".png")
        image = Image.open(image_path)
        tensor = trans(image)  # (C,W,H)

        enter_point = torch.tensor(seq[0], dtype=torch.float)  # dim = 2
        esc_point = torch.tensor(seq[-1], dtype=torch.float)  # dim = 2

        # <----- data preprocess ------->
        # pay attetion! seq_inv differs in the following two situation
        if seq_len <= self.max_len:
            seq_inv = seq[::-1]
            seq += [[0.0, 0.0] for _ in range(self.max_len - seq_len)]
            seq_inv += [[0.0, 0.0] for _ in range(self.max_len - seq_len)]
            seq_inv = torch.tensor(seq_inv, dtype=torch.float)
        elif seq_len > self.max_len:
            # systematically sample a subset for long sequence
            dis = int(seq_len / self.max_len)
            ind = [dis * i for i in range(self.max_len)]
            seq = [seq[i] for i in ind]
            seq_inv = torch.tensor(
                seq[::-1], dtype=torch.float
            )  # (max_len,2) inverse orde
            seq_len = self.max_len  # be careful!
        # <----------------------------->

        seq = torch.tensor(seq, dtype=torch.float)  # (max_len, 2)
        seq_len = torch.tensor(seq_len, dtype=torch.long).unsqueeze(0)

        return {
            "name": image_name,
            "image": tensor,
            "seq": seq,
            "seq_inv": seq_inv,
            "enter": enter_point,
            "esc": esc_point,
            "len": seq_len,
        }

    def train_set(self):
        """call this method to switch to train mode"""
        self.data = self.train_data
        return copy.deepcopy(self)

    def test_set(self):
        """call this method to switch to test mode"""
        self.data = self.test_data
        return copy.deepcopy(self)


def intervals_avg(seq):
    """used to calculate the average intervals of a certain sequence"""
    # seq = seq.tolist()
    seq_ = np.array(seq[1:])
    seq = np.array(seq[:-1])
    intervals = np.sqrt(np.power(seq - seq_, 2).sum(axis=1)).mean()
    return intervals


def cal_dis(seq):
    """calculate the distance between seq[0] and seq[-1]"""
    x1 = seq[0][0]
    y1 = seq[0][1]
    x2 = seq[-1][0]
    y2 = seq[-1][1]
    dis = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2).item()
    return dis