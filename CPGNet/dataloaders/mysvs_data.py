import torch
import dgl
import os
import numpy as np
import pickle as pkl
import torch.utils.data as data
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from dgl.data.utils import load_graphs
from sklearn.utils import shuffle
import random


class MysvsData(data.Dataset):
    def __init__(
        self,
        data_dir="",
        dataset="",
        split="",
        label_file="",
        text_file="",
        train="train",
        kfold=0,
        fold_num=0,
        seed=3407,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.file_list = [
            os.path.join(self.data_dir, path) for path in os.listdir(self.data_dir)
        ]
        self.label_dict = pd.read_csv(self.label_file)
        if text_file is not None:
            with open(text_file, "rb") as f:
                self.text_feature = pkl.load(f)
        else:
            self.text_feature = None
        self.check_files()

    def check_files(self):
        # 加载数据列表
        # file_list = shuffle(self.file_list, random_state=3407)
        # fl_train_val, fl_test = train_test_split(
        #     file_list, test_size=0.1, random_state=42
        # )
        # if self.kfold > 1:
        #     kf = KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        #     fl_train_idx, fl_val_idx = list(kf.split(fl_train_val))[self.fold_num]
        #     fl_train = np.array(fl_train_val)[fl_train_idx]
        #     fl_val = np.array(fl_train_val)[fl_val_idx]
        # else:
        #     fl_train, fl_val = train_test_split(
        #         fl_train_val, test_size=0.111, random_state=42
        #     )
        fl_train = (
            pd.read_csv(os.path.join(self.split, "train.txt"), header=None)
            .sample(frac=1)
            .values.tolist()[0]
        )
        fl_val = (
            pd.read_csv(os.path.join(self.split, "val.txt"), header=None)
            .sample(frac=1)
            .values.tolist()[0]
        )
        fl_test = (
            pd.read_csv(os.path.join(self.split, "test.txt"), header=None)
            .sample(frac=1)
            .values.tolist()[0]
        )
        if self.train == "train":
            self.path_list = fl_train
        elif self.train == "val":
            self.path_list = fl_val
        elif self.train == "test":
            self.path_list = fl_test
        elif self.train == "predict":
            # self.path_list = fl_test+fl_val+fl_train
            self.path_list = ["23-26644-2#1_6", "23-55788-2#6_11"]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # path = self.path_list[idx]
        # g_name = os.path.splitext(os.path.basename(path))[0]
        path = os.path.join(self.data_dir, self.path_list[idx] + ".bin")
        g_name = os.path.splitext(os.path.basename(path))[0]
        g, _ = load_graphs(path)
        label = self.label_dict.loc[
            self.label_dict.slide_id == g_name + ".svs"
        ].values.tolist()
        g = g[0]
        # g = dgl.add_self_loop(g,etype=('0', '4', '0'))
        g = dgl.add_self_loop(g)
        if self.text_feature is not None:
            return g, self.text_feature, torch.FloatTensor(label[0][1:]), g_name
        else:
            return g, torch.FloatTensor(label[0][1:]), g_name
