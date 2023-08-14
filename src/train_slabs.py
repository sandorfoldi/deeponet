import torch 
import argparse
from model import DeepONet, DeepONet1DCNN, DeepONet2DCNN
from wave_loader import get_wave_datasets
import numpy as np
import os
from glob import glob
import wandb

from softadapt import SoftAdapt
from slabs import SlabWeights
"""
    Class used to train and save a DeepONet model:
    Input: 


    Output:
    
"""
def make_loss_col_wave_eq(c):
    def loss_col(net, u, xt):
        xt.requires_grad = True
        u.requires_grad = True
        preds = net(u, xt)
        # print(u.requires_grad)
        # print(preds.requires_grad)
        # print(xt.requires_grad)
        # first time derivative of preds
        ddt_preds = torch.autograd.grad(preds, xt, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]
        # second time derivative of preds
        d2dt2_preds = torch.autograd.grad(ddt_preds[:, 0], xt, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]

        # first x derivative of preds
        ddx_preds = torch.autograd.grad(preds, u, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]
        # second x derivative of preds
        d2dx2_preds = torch.autograd.grad(ddx_preds[:, 0], u, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]

        return torch.nn.MSELoss()(d2dt2_preds[:, 0], c**2 * d2dx2_preds[:, 0])
    return loss_col



def save_model(model, root, foldername):
    if not os.path.exists(root + '/models/' + foldername):
        os.makedirs(root + '/models/' + foldername)

    filename = str(model) + '.pt'
    torch.save(model.state_dict(), root + '/models/' + foldername + '/' + filename)


def save_results(losses, root, foldername, filename):
    if not os.path.exists(root + '/results/' + foldername):
        os.makedirs(root + '/results/' + foldername)

    np.save(root + '/results/' + foldername + '/' + filename, losses)

def train_model(args):
    dataset_path = args.dataset
    n_points = args.n_points
    model_name = args.model
    hidden_units = args.n_hidden
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    run_name = args.run_name

    # login to wandb
    # api key is stored in private/wandb_api_key.txt
    wandb.login(key=open('private/wandb_api_key.txt').read().strip())

    # start a new wandb run to track this script
    wandb.init(
        entity="sandorfoldi",
        # set the wandb project where this run will be logged
        project="deeponet",
        # name
        name=run_name,
        # track hyperparameters and run metadata
        config={
        "dataset_path": dataset_path,
        "n_points": n_points,
        "model_name": model_name,
        "hidden_units": hidden_units,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device:\t{device}')
    
    root = os.getcwd()

    paths = glob(dataset_path +"/" + '*.npy')

    if (len(paths) == 0):
        raise Exception('No files found in dataset folder')
    
    # Load dataset
    ds_train, ds_valid = get_wave_datasets(paths, n_points=n_points, device=device)

    # Train and validation loaders
    train_dataloader =  torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    # Model
    model = DeepONet(100, hidden_units, hidden_units)

    if model_name == 'CNN1D':
        model = DeepONet1DCNN(100, hidden_units, hidden_units)
    elif model_name == 'CNN2D':
        model = DeepONet2DCNN(100, hidden_units, hidden_units)

    model.to(device)

    # asserting model and dataset devices
    model_device = next(model.parameters()).device
    sample = next(iter(train_dataloader))
    x_device, y_device, u_device = sample[0].device, sample[1].device, sample[2].device
    
    assert model_device == device, "model not on correct device"
    assert x_device == device, "x not on correct device"
    assert y_device == device, "y not on correct device"
    assert u_device == device, "u not on correct device"

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # Train
    epoch_train_losses, epoch_val_losses = [], []

    loss_col = make_loss_col_wave_eq(5)

    print(f'Training {str(model)} for {epochs} epochs')    
    
    slab_weights = SlabWeights(
        num_x_slabs=10,
        num_t_slabs=10,
        x_min=ds_train.xmin,
        x_max=ds_train.xmax,
        t_max=ds_train.tmax
    )

    softadapt = SoftAdapt(1 + slab_weights.num_weights, beta=0.1)

    for epoch in range(epochs):
        # wandb.log({"epoch": epoch})
        # Training
        model.train()
        train_losses = []

        for (xt_batch, y_batch, u_batch) in train_dataloader:
            # update softadapt coefficients
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass through network
            pred = model(u_batch, xt_batch)

            # loss_boundary = loss_fn(pred, y_batch.view(-1))
            loss_boundary = torch.pow((pred - y_batch.view(-1)), torch.tensor(2.))
            loss_collocation = loss_col(model, u_batch, xt_batch)

            alpha_collocation = softadapt.get_alphas()[0]
            alphas_slabs = slab_weights.get_at_xt(xt_batch)

            loss = alpha_collocation * loss_collocation + torch.sum(alphas_slabs * loss_boundary)
            
            with torch.no_grad():
                softadapt.update(torch.cat([loss_collocation.view(1), slab_weights.tranform_loss(loss_boundary, xt_batch)]))

                # update slabs weightes
                slab_weights.update(softadapt.get_alphas()[1:])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            wandb.log({"train_loss_slabs_mean": torch.mean(loss_boundary).item(), "epoch": epoch})
            wandb.log({"train_loss_collocation": loss_collocation.item(), "epoch": epoch})
            wandb.log({"alpha_slabs_mean": torch.mean(alphas_slabs).item(), "epoch": epoch})
                
        epoch_train_losses.append(np.mean(train_losses))
        wandb.log({"epoch_train_loss": epoch_train_losses[-1], "epoch": epoch})
        lr_scheduler.step()



        # Validation
        model.eval()
        validation_losses = []
        model.eval()
        for (xt_batch, y_batch, u_batch) in validation_dataloader:

            validation_losses.append(loss.item())
    
        epoch_val_losses.append(np.mean(validation_losses))
        wandb.log({"epoch_valid_loss": epoch_val_losses[-1], "epoch": epoch})
        
        # print train and validation losses. Format 6 decimals
        print(f'Epoch {epoch+1}/{epochs} - Train loss: {epoch_train_losses[-1]:.6f} - Validation loss: {epoch_val_losses[-1]:.6f}')
        

    return model, epoch_train_losses, epoch_val_losses


def loss_col_fac(rhs, loss=torch.nn.MSELoss()):
    def loss_col(net, u, xt, y):
        preds = net(u, xt)
        dpreds = torch.autograd.grad(preds, xt, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True)[0]
        dtarget = rhs(xt[:, 0], y, xt[:, 1])
        return loss(dtarget, dpreds[:, 0])
    return loss_col


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='data/test_1a')
    args.add_argument('--model', type=str, default='FFNN')
    args.add_argument('--n_hidden', type=int, default=128)
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=1e-3) 
    args.add_argument('--n_points', type=int, default=128)
    args.add_argument('--outputfolder', type=str, default='default')
    args.add_argument('--run_name', type=str, default='default')
    args.add_argument('--beta', type=float, default=0.1)
    args = args.parse_args()

    root = os.getcwd()

    model, train_losses, val_losses = train_model(args)

    output_folder = args.outputfolder
    
    print(f'Saving model and results to {output_folder}')

    save_model(model, root, output_folder)
    save_results(train_losses, root, output_folder, 'train_losses')
    save_results(val_losses, root, output_folder, 'val_losses')




