import torch
import numpy as np
from glob import glob


def get_wave_datasets(paths, device='cpu', splits = (0.8, 0.2)):
    # load and shuffle paths
    paths = np.array(paths)
    np.random.shuffle(paths)

    # create and shuffle x (place) and t (time) indices
    x_idxs = np.array(range(100))
    t_idxs = np.array(range(100))

    np.random.shuffle(x_idxs)
    np.random.shuffle(t_idxs)

    xt_idxs = np.array([(x, t) for x in x_idxs for t in t_idxs])

    # split paths, x_idxs, and t_idxs into train and val
    assert len(splits) == 2 and sum(splits) == 1, 'splits must be a tuple of length 2 and sum to 1'

    train_paths = paths[:int(len(paths) * splits[0])]
    val_paths = paths[int(len(paths) * splits[0]):]

    train_xt_idxs = xt_idxs[:int(len(xt_idxs) * splits[0])].reshape([-1, 100, 2])
    val_xt_idxs = xt_idxs[int(len(xt_idxs) * splits[0]):].reshape([-1, 100, 2])

    # load train data
    train_xts = []
    train_ys = []
    train_us = []

    assert len(train_paths) == len(train_xt_idxs), 'train_paths and train_xt_idxs must be the same length'
    for path, xt_idx in zip(train_paths, train_xt_idxs):
        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        train_xts.append(np.stack([x[xt_idx[:, 0]], t[xt_idx[:, 1]]], axis=1))
        train_ys.append(y[xt_idx[:, 0], xt_idx[:, 1]])
        train_us.append([u]*len(xt_idx))

    train_xts = np.array(train_xts).reshape([-1, 2])
    train_ys = np.array(train_ys).reshape([-1, 1])
    train_us = np.array(train_us).reshape([-1, 10])

    
    # load val data
    val_xts = []
    val_ys = []
    val_us = []

    assert len(val_paths) == len(val_xt_idxs), 'val_paths and val_xt_idxs must be the same length'
    for path, xt_idx in zip(val_paths, val_xt_idxs):
        data = np.load(path, allow_pickle=True)
        x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

        val_xts.append(np.stack([x[xt_idx[:, 0]], t[xt_idx[:, 1]]], axis=1))
        val_ys.append(y[xt_idx[:, 0], xt_idx[:, 1]])
        val_us.append([u]*len(xt_idx))
    
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
    paths = glob('data/a/*.npy')
    ds_train, ds_valid = get_wave_datasets(paths)
    for i in range(10):
        xt, y, u = ds_train[i]
        pass
