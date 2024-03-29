import torch
import numpy as np
from glob import glob
import math


def get_wave_datasets(paths, splits = (0.8, 0.2), n_points=100, device='cpu'):
    # load and shuffle paths
    paths = np.array(paths)
    np.random.shuffle(paths)

    n_ic = len(paths)
    
    # load the first array to get the shape
    data = np.load(paths[0], allow_pickle=True)
    x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

    xmin, xmax = x.min(), x.max()
    tmax = t.max()

    n_sensors = len(u)

    x_all_idxs = np.arange(len(x))
    t_all_idxs = np.arange(len(t))
    ic_all_idxs = np.arange(n_ic)

    
    ic_train_idxs = ic_all_idxs[:int(n_ic*splits[0])]
    ic_val_idxs = ic_all_idxs[int(n_ic*splits[0]):]

    train_xts = np.zeros([0, 2])
    train_ys = np.zeros([0, 1])
    train_us = np.zeros([0, n_sensors])

    val_xts = np.zeros([0, 2])
    val_ys = np.zeros([0, 1])
    val_us = np.zeros([0, n_sensors])

    for ic in ic_train_idxs:
        path = paths[ic]

        # create random indeces per ic
        xt_train_idxs = np.array([(x, t) for x in x_all_idxs for t in t_all_idxs])
        select_train_idxs = np.random.choice(np.arange(xt_train_idxs.shape[0]), n_points, replace=False)
        xt_train_idxs = xt_train_idxs[select_train_idxs,:]

        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        train_xts = np.concatenate([train_xts, np.stack([x[xt_train_idxs[:, 0]], t[xt_train_idxs[:, 1]]], axis=1)], axis=0)
        train_ys = np.concatenate([train_ys, (y[xt_train_idxs[:, 0], xt_train_idxs[:, 1]]).reshape([-1, 1])], axis=0)
        train_us = np.concatenate([train_us, np.concatenate([u.reshape([1, -1])]*n_points, axis=0)], axis=0)

    for ic in ic_val_idxs:
        path = paths[ic]

        # create random indeces per ic
        xt_val_idxs = np.array([(x, t) for x in x_all_idxs for t in t_all_idxs])
        select_val_idxs = np.random.choice(np.arange(xt_val_idxs.shape[0]), n_points, replace=False)
        xt_val_idxs = xt_val_idxs[select_val_idxs,:]

        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        val_xts = np.concatenate([val_xts, np.stack([x[xt_val_idxs[:, 0]], t[xt_val_idxs[:, 1]]], axis=1)], axis=0)
        val_ys = np.concatenate([val_ys, (y[xt_val_idxs[:, 0], xt_val_idxs[:, 1]]).reshape([-1, 1])], axis=0)
        val_us = np.concatenate([val_us, np.concatenate([u.reshape([1, -1])]*n_points, axis=0)], axis=0)
    
    assert set([tuple(i) for i in train_us]).intersection(set([tuple(i) for i in val_us])) == set(), 'same initial condition present in train and val set'

    ds_train = WaveDataset(train_xts, train_ys, train_us, xmin, xmax, tmax, device=device)
    ds_valid = WaveDataset(val_xts, val_ys, val_us, xmin, xmax, tmax, device=device)

    return ds_train, ds_valid


class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, xts, ys, us, xmin, xmax, tmax, device='cpu', dtype=torch.float32):
        self.xts = torch.tensor(xts, dtype=dtype, device=device)
        self.ys = torch.tensor(ys, dtype=dtype, device=device)
        self.us = torch.tensor(us, dtype=dtype, device=device)
        self.xmin = xmin
        self.xmax = xmax
        self.tmax = tmax


    def __len__(self):
        return len(self.xts)
    
    def __getitem__(self, idx):
        return self.xts[idx], self.ys[idx], self.us[idx]


if __name__ == '__main__':
    paths = glob('data/training/*.npy')
    ds_train, ds_valid = get_wave_datasets(paths)
    dl_train, dl_valid = torch.utils.data.DataLoader(ds_train, batch_size=32), torch.utils.data.DataLoader(ds_valid, batch_size=32)
    for x, y, u in dl_train:
        print(x.shape, y.shape, u.shape)
        break
