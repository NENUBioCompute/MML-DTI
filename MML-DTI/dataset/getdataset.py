import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class Mydataset(Dataset):
    def __init__(self, samples, compounds, proteins):
        self.samples = samples
        self.len = len(samples)
        self.compounds = compounds
        self.proteins = proteins

    def __getitem__(self, idx):
        one_sample = self.samples[idx]
        return self.compounds[int(one_sample[0])], self.proteins[int(one_sample[1])], torch.tensor(
            [one_sample[2]]), one_sample

    def __len__(self):
        return self.len


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


def load_tensor(file_name, dtype):
    data = np.load(file_name + '.npy', allow_pickle=True)
    # 将数据转换为 numpy.float32 类型
    data = [d.astype(np.float32) for d in data]
    return [dtype(d) for d in data]


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def preparedata(type, dataset, fold=0):
    dir_input = ('dataset/' + dataset + '/')
    compounds = load_tensor(dir_input + 'smilesembeddings', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteinsembeddings-base', torch.FloatTensor)

    # 修正路径格式
    datadir_input = f'data/{dataset}/{type}/fold{fold}/'

    trainfiles = pd.read_csv(datadir_input + 'train/samples.csv')
    validfiles = pd.read_csv(datadir_input + 'valid/samples.csv')
    testfiles = pd.read_csv(datadir_input + 'test/samples.csv')

    return trainfiles.values, validfiles.values, testfiles.values, compounds, proteins


def collatef(batch):
    batchlist = []
    for item in batch:
        compound, protein, interaction, sample = item
        list = [compound, protein, interaction, sample]
        batchlist.append(list)
    return batchlist


def preparedataset(batch_size, type, dataset, fold=0):
    trainsamples, validsamples, testsamples, compounds, proteins = preparedata(type, dataset, fold)

    trainloader = DataLoader(Mydataset(trainsamples, compounds, proteins), shuffle=True, batch_size=batch_size,
                             collate_fn=collatef, drop_last=False)
    validloader = DataLoader(Mydataset(validsamples, compounds, proteins), shuffle=False, batch_size=batch_size,
                             collate_fn=collatef, drop_last=False)
    testloader = DataLoader(Mydataset(testsamples, compounds, proteins), shuffle=False, batch_size=batch_size,
                            collate_fn=collatef, drop_last=False)
    return trainloader, validloader, testloader, compounds, proteins