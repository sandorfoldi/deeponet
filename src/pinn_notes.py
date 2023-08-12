import torch
import numpy as np

rhs_torch = lambda t, y, a: a*torch.cos(a*t)
rhs_numpy = lambda t, y, a: a*np.cos(a*t)


def wave_equation(t, y, c):
    u, v = np.split(y, 2)
    d2udx2 = np.gradient(np.gradient(u))
    v[0] = 0
    v[-1] = 0
    return np.concatenate((v, c**2 * d2udx2))


def loss_col_fac(rhs, params, loss=torch.nn.MSELoss()):
    def loss_col(net, t, y):
        preds = net(t)
        dpreds = torch.autograd.grad(preds, t, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]
        dtarget = rhs(t[:, 0], y, params)
        return loss(dtarget, dpreds[:, 0])
    return loss_col




loss_col = loss_col_fac(rhs_torch)


# loss_c = loss_col(model, x_boundary, t_boundary)


