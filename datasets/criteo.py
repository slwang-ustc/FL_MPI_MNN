import MNN
import mnist
import numpy as np
import pandas as pd
from config import cfg

F = MNN.expr


class RecDataset(MNN.data.Dataset):
    def __init__(self, training_dataset=True):
        super(RecDataset, self).__init__()
        self.is_training_dataset = training_dataset
        data_set = pd.read_table(cfg['dataset_path'], header=None)
        data = data_set[range(39, 78)]
        labels = data_set[78]
        if self.is_training_dataset:
            self.data = np.array(data)[:168000]
            self.labels = np.array(labels)[:168000]
        else:
            self.data = np.array(data)[168000:]
            self.labels = np.array(labels)[168000:]

    def __getitem__(self, index):
        dv = F.const(self.data[index].flatten().tolist(), [39], F.data_format.NCHW)
        dl = F.const([self.labels[index]], [], F.data_format.NCHW, F.dtype.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return len(self.data)
        else:
            return len(self.data)
