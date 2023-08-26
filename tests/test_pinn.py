from glob import glob

import torch
from torch.nn import Module
from src.pinn import make_loss_col_wave_eq
from src.wave_loader import get_wave_datasets

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
    # compare to finite difference on simple simulation
    paths = glob('/work3/s216416/deeponet/data/test_1a/*')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train, ds_valid = get_wave_datasets(paths, n_points=10, device=device)

    # Train and validation loaders
    train_dataloader =  torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)

    xt_batch, y_batch, u_batch = next(iter(train_dataloader))

    loss_col = make_loss_col_wave_eq(5)
    net = Pinn0()

    d2dt2_preds, d2dx2_preds, loss = loss_col(net, u_batch, xt_batch, debug=True)


def todo_test_pinn_3():
    pass


if __name__ == "__main__":
    test_pinn_1()
