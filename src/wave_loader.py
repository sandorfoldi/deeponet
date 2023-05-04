import torch
import numpy as np


class WaveLoader(torch.utils.data.IterableDataset):
    def __init__(self, paths, params,):
        super(WaveLoader, self).__init__()
        self.params = params
        self.data = []
        for path in paths:
            data = np.load(path, allow_pickle=True)
            self.data.append({'x': data['x'], 't': data['t'], 'y': data['y'], 'u': data['u'],})
    
    @staticmethod
    def random_item(arr):
        return arr[np.random.randint(0, len(arr))]


    def __next__(self):
        # choose a random data point should be random
        idx_sim = np.random.randint(0, len(self.data))
        sim  = self.data[idx_sim]
        
        idx_x = np.random.randint(0, len(sim['x']))
        idx_t = np.random.randint(0, len(sim['t']))

        x = sim['x'][0][idx_x]
        t = sim['t'][0][idx_t]
        y = sim['y'][0][idx_x, idx_t]
        u = sim['u'][0]

        return x, t, y, u

if __name__ == '__main__':
    loader = WaveLoader(['data/0.npy'], {})
    for i in range(10):
        x, t, y, u = next(loader)
        pass
