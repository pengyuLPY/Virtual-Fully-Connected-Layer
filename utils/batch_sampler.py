import numpy as np
from tqdm import tqdm
import random
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    __module__ = __name__
    __qualname__ = 'BalancedBatchSampler'

    def __init__(self, dataset, n_samples, n_min_samples, batch_size, shuffle):
        self.dataset = dataset
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label:np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
        self.indices_random = np.array([i for i in range(len(self.labels))])
        self.used_indice_bool = np.array([0 for i in range(len(self.labels))])
        self.used_label_indices_count = {label:0 for label in self.labels_set}
        if shuffle:
            np.random.shuffle(self.indices_random)
            for l in self.labels_set:
                np.random.shuffle(self.label_to_indices[l])

        self.n_samples = n_samples
        self.n_min_samples = n_min_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert self.n_samples <= self.batch_size
        self.len_ = len(self.labels) // self.batch_size

    def __iter__(self):
        self.indices_random = np.array([i for i in range(len(self.labels))])
        self.used_indice_bool = np.array([0 for i in range(len(self.labels))])
        self.used_label_indices_count = {label:0 for label in self.labels_set}
        if self.shuffle:
            np.random.shuffle(self.indices_random)
            for l in self.labels_set:
                np.random.shuffle(self.label_to_indices[l])

        indices_cur = []
        for ind in tqdm(self.indices_random):
            if self.used_indice_bool[ind] != 0:
                continue
            indices_cur.append(ind)
            self.used_indice_bool[ind] = 1
            class_ = self.labels.numpy()[ind]
            number = 1
            sta_ind = self.used_label_indices_count[class_]
            for i in range(sta_ind, len(self.label_to_indices[class_])):
                ind_2 = self.label_to_indices[class_][i]
                if self.used_indice_bool[ind_2] == 0:
                    indices_cur.append(int(ind_2))
                    self.used_indice_bool[ind_2] = 1
                    number += 1
                self.used_label_indices_count[class_] = i + 1
                if number >= self.n_samples:
                    break

            while number < self.n_min_samples:
                ind_2 = np.random.choice((self.label_to_indices[class_]), 1, replace=False)
                indices_cur.append(int(ind_2))
                self.used_indice_bool[ind_2] = 1
                number += 1

        len_cur = 0
        for i in range(0, len(indices_cur) - self.batch_size, self.batch_size):
            len_cur += 1
            yield indices_cur[i:i + self.batch_size]

        self.len_ = len_cur

    def __len__(self):
        return self.len_
