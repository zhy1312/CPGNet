import torch
import dgl
import os
import pickle as pkl
import torch.utils.data as data
import pandas as pd
from dgl.data.utils import load_graphs


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset="",
        text_file="",
        train="train",
        seed=3407,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        data_dir = os.path.join("graphs", self.dataset, "features")
        label_file = os.path.join("labels", self.dataset)
        self.file_list = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
        self.label_dict = pd.read_csv(os.path.join(label_file, "labels.csv"))
        with open(os.path.join(label_file, "text_embedings.pkl"), "rb") as f:
            self.text_feature = pkl.load(f)

        self.data_dir = data_dir
        self.label_file = label_file

    def check_files(self):
        fl_train = (
            pd.read_csv(os.path.join(self.label_file, "train.txt"), header=None)
            .sample(frac=1)
            .values.tolist()[0]
        )
        fl_val = (
            pd.read_csv(os.path.join(self.label_file, "val.txt"), header=None)
            .sample(frac=1)
            .values.tolist()[0]
        )
        fl_test = (
            pd.read_csv(os.path.join(self.label_file, "test.txt"), header=None)
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
            self.path_list = fl_test + fl_val + fl_train

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.path_list[idx] + ".bin")
        g_name = os.path.splitext(os.path.basename(path))[0]
        g, _ = load_graphs(path)
        label = self.label_dict.loc[self.label_dict.slide_id == g_name].values.tolist()
        g = g[0]
        g = dgl.add_self_loop(g)
        if self.text_feature is not None:
            return g, self.text_feature, torch.FloatTensor(label[0][1:]), g_name
        else:
            return g, torch.FloatTensor(label[0][1:]), g_name
