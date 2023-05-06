import torch
import numpy as np
from glob import glob
import math



def get_wave_datasets(paths, splits = (0.8, 0.2), n_points=100):
    # load and shuffle paths
    paths = np.array(paths)
    np.random.shuffle(paths)

    n_ic = len(paths)
    
    
    # load the first array to get the shape
    data = np.load(paths[0], allow_pickle=True)
    x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

    
    x_all_idxs = np.arange(len(x))
    t_all_idxs = np.arange(len(t))
    ic_all_idxs = np.arange(n_ic)

    np.random.shuffle(ic_all_idxs)
    ic_train_idxs = ic_all_idxs[:int(n_ic*splits[0])]
    ic_val_idxs = ic_all_idxs[int(n_ic*splits[0]):]

    xtic_train_idxs = np.array([(ic, x, t) for x in x_all_idxs for t in t_all_idxs for ic in ic_train_idxs])
    xtic_val_idxs = np.array([(ic, x, t) for x in x_all_idxs for t in t_all_idxs for ic in ic_val_idxs])

    # randomly select n_points from them
    select_train_idxs = np.random.choice(np.arange(xtic_train_idxs.shape[0]), n_points*n_ic, replace=False)
    select_val_idxs = np.random.choice(np.arange(xtic_val_idxs.shape[0]), n_points*n_ic, replace=False)

    xtic_train_idxs = xtic_train_idxs[select_train_idxs,:]
    xtic_val_idxs = xtic_val_idxs[select_val_idxs,:]

    # test data drifting in initial conditions
    train_ics = set([xtic[0] for xtic in xtic_train_idxs])
    val_ics = set([xtic[0] for xtic in xtic_val_idxs])

    assert train_ics.intersection(val_ics) == set(), 'same inital condition present in train and val'
    

    # load train data
    train_xts = []
    train_ys = []
    train_us = []

    for xtic in xtic_train_idxs:
        path = paths[xtic[0]]
        xt_idx = xtic[1:]

        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        train_xts.append([x[xt_idx[0]], t[xt_idx[1]]])
        train_ys.append(y[xt_idx[0], xt_idx[1]])
        train_us.append(u)

    train_xts = np.array(train_xts).reshape([-1, 2])
    train_ys = np.array(train_ys).reshape([-1, 1])
    train_us = np.array(train_us).reshape([-1, 10])

    
    # load val data
    val_xts = []
    val_ys = []
    val_us = []

    for xtic in xtic_val_idxs:
        path = paths[xtic[0]]
        xt_idx = xtic[1:]

        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        val_xts.append([x[xt_idx[0]], t[xt_idx[1]]])
        val_ys.append(y[xt_idx[0], xt_idx[1]])
        val_us.append(u)
    
    val_xts = np.array(val_xts).reshape([-1, 2])
    val_ys = np.array(val_ys).reshape([-1, 1])
    val_us = np.array(val_us).reshape([-1, 10])

    # create and return dataloaders
    train_loader = WaveDataset(train_xts, train_ys, train_us)
    val_loader = WaveDataset(val_xts, val_ys, val_us)

    return train_loader, val_loader


class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, xts, ys, us, device='cpu', dtype=torch.float32):
        self.xts = torch.tensor(xts, dtype=dtype, device=device)
        self.ys = torch.tensor(ys, dtype=dtype, device=device)
        self.us = torch.tensor(us, dtype=dtype, device=device)

    def __len__(self):
        return len(self.xts)
    
    def __getitem__(self, idx):
        return self.xts[idx], self.ys[idx], self.us[idx]


if __name__ == '__main__':
    paths = glob('data/10/*.npy')
    ds_train, ds_valid = get_wave_datasets(paths)
    for i in range(10):
        xt, y, u = ds_train[i]
        pass
    