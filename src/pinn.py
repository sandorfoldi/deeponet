import torch


def make_loss_col_wave_eq(c):
    def loss_col(net, u, xt, debug=False):
        xt.requires_grad = True
        u.requires_grad = True
        preds = net(u, xt)

        
        # first xt derivative of preds
        ddxt_preds = torch.autograd.grad(preds, xt, create_graph=True, grad_outputs=torch.ones_like(preds))[0]

        # second x derivative of preds
        d2dx2_preds = torch.autograd.grad(ddxt_preds[:, 0], xt, create_graph=True, grad_outputs=torch.ones_like(ddxt_preds[:, 0]))[0][0, 0]

        # second t derivative of preds
        d2dt2_preds = torch.autograd.grad(ddxt_preds[:, 1], xt, create_graph=True, grad_outputs=torch.ones_like(ddxt_preds[:, 1]))[0][0, 1]


        if debug:
            return d2dt2_preds, d2dx2_preds, torch.nn.MSELoss()(d2dt2_preds, c**2 * d2dx2_preds)
        else:
            return torch.nn.MSELoss()(d2dt2_preds, c**2 * d2dx2_preds)


    return loss_col
