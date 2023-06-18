import random
import numpy as np
from MNN.data import DataLoader
from config import cfg
import datasets.mnist
import datasets.criteo
import datasets.avazu
import MNN


class Partition(MNN.data.Dataset):
    def __init__(self, data, index):
        super().__init__()
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs

    def __len__(self):
        return len(self.data)


class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        # for class_idx in range(len(data.classes)):
        for class_idx in range(cfg['classes_size']):
            label_indexes.append(list(np.where(np.array(data.labels) == class_idx)[0]))
            class_len.append(len(label_indexes[class_idx]))
            rng.shuffle(label_indexes[class_idx])

        # distribute class indexes to each vm according to sizes matrix
        for class_idx in range(cfg['classes_size']):
            begin_idx = 0
            for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                end_idx = begin_idx + round(frac * class_len[class_idx])
                end_idx = int(end_idx)
                self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs

    def __len__(self):
        return len(self.data)


def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True):
    if selected_idxs is None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def load_datasets(dataset_type, data_path=cfg['dataset_path']):
    if dataset_type == 'MNIST_MNN':
        train_dataset = datasets.mnist.MnistDataset(training_dataset=True)
        test_dataset = datasets.mnist.MnistDataset(training_dataset=False)
    elif dataset_type == 'criteo':
        train_dataset = datasets.criteo.RecDataset(training_dataset=True)
        test_dataset = datasets.criteo.RecDataset(training_dataset=False)
    elif dataset_type == 'avazu':
        train_dataset = datasets.avazu.AvazuDataset(is_training=True)
        test_dataset = datasets.avazu.AvazuDataset(is_training=False)
    else:
        raise ValueError("Not valid dataset type")

    return train_dataset, test_dataset