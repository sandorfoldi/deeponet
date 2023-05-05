import torch
import numpy as np
from glob import glob


class WaveLoader(torch.utils.data.Dataset):
    def __init__(self, paths, params, device='cpu', splits = (0.8, 0.2)):
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
        self.train_xts = []
        self.train_ys = []
        self.train_us = []

        assert len(train_paths) == len(train_xt_idxs), 'train_paths and train_xt_idxs must be the same length'
        for path, xt_idx in zip(train_paths, train_xt_idxs):
            data = np.load(path, allow_pickle=True)
            x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

            self.train_xts.append(np.stack([x[xt_idx[:, 0]], t[xt_idx[:, 1]]], axis=1))
            self.train_ys.append(y[xt_idx[:, 0], xt_idx[:, 1]])
            self.train_us.append([u]*len(xt_idx))

        self.train_xts = np.array(self.train_xts).reshape([-1, 2])
        self.train_ys = np.array(self.train_ys).reshape([-1, 1])
        self.train_us = np.array(self.train_us).reshape([-1, 10])

        
        # load val data
        self.val_xts = []
        self.val_ys = []
        self.val_us = []

        assert len(val_paths) == len(val_xt_idxs), 'val_paths and val_xt_idxs must be the same length'
        for path, xt_idx in zip(val_paths, val_xt_idxs):
            data = np.load(path, allow_pickle=True)
            x, t, y, u = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

            self.val_xts.append(np.stack([x[xt_idx[:, 0]], t[xt_idx[:, 1]]], axis=1))
            self.val_ys.append(y[xt_idx[:, 0], xt_idx[:, 1]])
            self.val_us.append([u]*len(xt_idx))
        
        self.val_xts = np.array(self.val_xts).reshape([-1, 2])
        self.val_ys = np.array(self.val_ys).reshape([-1, 1])
        self.val_us = np.array(self.val_us).reshape([-1, 10])

        
    def __len__(self):
        return len(self.train_xts) + len(self.val_xts)
    
    def __getitem__(self, idx):


        










    def __next__(self):
        pass

if __name__ == '__main__':
    paths = glob('data/*.npy')
    loader = WaveLoader(paths, {})
    for i in range(10):
        x, t, y, u = next(loader)
        pass
