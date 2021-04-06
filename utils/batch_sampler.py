import numpy as np
from tqdm import tqdm
import random
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler_LPY(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, ratio=None):
        self.labels = dataset.labels

        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        random.shuffle(self.labels_set)
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.finished_label_sign = {label: False for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes


        self.len_ = len(self.labels) // self.batch_size
        self.ratio_ = ratio
      

    def __iter__(self):

        for class_ in self.labels_set:
             #np.random.shuffle(self.label_to_indices[class_])
             #self.used_label_indices_count[class_] = 0
             self.finished_label_sign[class_] = False

      
        indices = []
        num_finished_class = 0
        class_sta = 0
        class_end = 0 #min(len(self.labels_set), self.n_classes)
        count_times = 0
        while num_finished_class < len(self.labels_set):
            if self.ratio_ is not None and len(indices) > self.ratio_*len(self.labels):
                print('generatring batch sample indices,', num_finished_class, 'vs',len(self.labels_set))
                break
            if count_times % 1000 == 0:
                print('generatring batch sample indices,', num_finished_class, 'vs',len(self.labels_set), 'vs', len(indices))
            count_times += 1
            class_sta = class_end
            class_end = min(len(self.labels_set), class_sta+self.n_classes)
            classes = self.labels_set[class_sta:class_end]

            for class_ in classes:
                indices_candidates = self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples].copy()
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                    self.finished_label_sign[class_] = True

                while len(indices_candidates) < self.n_samples:
                    indices_candidates = np.append(indices_candidates, 
                               self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples].copy()
                        ) 
                    self.used_label_indices_count[class_] += self.n_samples
                    if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
                        self.finished_label_sign[class_] = True


                indices.extend(indices_candidates[:self.n_samples])

            num_finished_class = 0
            for key in self.finished_label_sign:
                if self.finished_label_sign[key]:
                    num_finished_class += 1

            class_end += 1
            if class_end > len(self.labels_set):
                class_end = 0
                random.shuffle(self.labels_set)
                for key in self.labels_set:
                    if self.finished_label_sign[key] == False:
                        break
                    class_end += 1
            #if indices.__len__() >= self.batch_size:
            #    yield indices[:self.batch_size]

        len_tmp = 0
        for i in range(0, len(indices)-self.batch_size, self.batch_size):
            len_tmp += 1
            yield indices[i:i+self.batch_size]

        self.len_ = len_tmp


    def __len__(self):
        return self.len_







class BalancedBatchSampler_LPY2(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        self.labelset2ratio = []
        summer_ = 0
        for label in self.labels_set:
            ratio = float(len(self.label_to_indices[label])) / len(self.labels) 
            summer_ += ratio
            #self.labelset2ratio.append(ratio)
            self.labelset2ratio.append( float(len(self.label_to_indices[label]))/len(self.labels) )
        print('BalancedBatchSampler_LPY2',len(self.labels), summer_)
        assert abs(summer_-1.0) < 1e-6, summer_

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

        self.ind_classes = 0

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False, p = self.labelset2ratio)
            indices = []
            for class_ in classes:
                indices_candidate = self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples]
                while len(indices_candidate) < self.n_samples:
                    indices_candidate = np.append(indices_candidate, indices_candidate)
                indices_candidate = indices_candidate[: self.n_samples]
                
                indices.extend(indices_candidate)

              
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size




class BalancedBatchSampler_LPY3(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    __module__ = __name__
    __qualname__ = 'BalancedBatchSampler_LPY3'

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
