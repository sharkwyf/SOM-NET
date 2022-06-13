import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import numpy as np


class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(np.stack(images), dtype=torch.float32).reshape((-1, 3, 32, 32)) / 255
        self.labels = torch.tensor(labels, dtype=torch.long).reshape((-1, 1))
        self.images = self.images[:]
        self.labels = self.labels[:]
        pass

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

 
def paired_collate_fn(batch):
    X = torch.stack([x for x, y in batch])
    Y = torch.stack([y for x, y in batch]).reshape(-1)
    return X, Y