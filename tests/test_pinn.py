from glob import glob
import numpy as np

import torch
from torch.nn import Module
from src.pinn import make_loss_col_wave_eq
from src.model import DeepONetNoBatchNorm

class Pinn0(Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, xt):
        return (u*u*xt*xt).sum()


def test_pinn_0():
    # output is not None

    loss_col = make_loss_col_wave_eq(5)
    assert loss_col is not None


def test_pinn_1():
    # autograd works as intended

    u = torch.tensor([[2., 3.]])
    xt = torch.tensor([[5., 7.]])
    loss_col = make_loss_col_wave_eq(5)

    net = Pinn0()

    d2dt2_preds, d2dx2_preds, loss = loss_col(net, u, xt, debug=True)
    
    assert d2dx2_preds == 2 * u[0][0]**2
    assert d2dt2_preds == 2 * u[0][1]**2


def todo_test_pinn_2():
    path = '/work3/s216416/deeponet/data/1g/800.npy'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(path, allow_pickle=True)
    xs, ts, ys, us = data['x'][0], data['t'][0], data['y'][0], data['u'][0]

    xs, ts, ys, us = torch.tensor([xs], device=device, dtype=torch.float32), torch.tensor([ts], device=device, dtype=torch.float32), torch.tensor([ys], device=device, dtype=torch.float32), torch.tensor([us], device=device, dtype=torch.float32)

    idx_x = 900
    idx_t = 100

    # finite difference second order derivatives
    d2y_dt2 = (ys[0][idx_x, idx_t+1] - 2 * ys[0][idx_x, idx_t] + ys[0][idx_x, idx_t-1]) / (ts[0][idx_t+1] - ts[0][idx_t-1])
    d2y_dx2 = (ys[0][idx_x+1, idx_t] - 2 * ys[0][idx_x, idx_t] + ys[0][idx_x-1, idx_t]) / (xs[0][idx_x+1] - xs[0][idx_x-1])

    print(f'd2y_dt2:\t\t\t{d2y_dt2:.4e}')
    print(f'd2y_dx2:\t\t\t{d2y_dx2:.4e}')
    print(f'd2y_dt2 - c^2 * d2y_dx2:\t{d2y_dt2 - 0.5**2 * d2y_dx2:.4e}')
    print('---------------------------------------')
    print('Estimating c based on finite difference')
    print(f'c = sqrt(d2y_dt2 / d2y_dx2) =\t{np.sqrt(d2y_dt2 / d2y_dx2):.4e}')



if __name__ == "__main__":
    todo_test_pinn_2()
