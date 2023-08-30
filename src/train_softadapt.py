import torch 
import argparse
from model import DeepONet, DeepONet1DCNN, DeepONet2DCNN
from wave_loader_timeslabs import get_wave_datasets_timeslabs
import numpy as np
import os
from glob import glob
import wandb
from softadapt import SoftAdapt
from src.pinn import make_loss_col_wave_eq

"""
    Class used to train and save a DeepONet model:
    Input: 


    Output:
    
"""


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
    mu_boundary = args.mu_boundary
    mu_colloc = args.mu_colloc
    mu_ic = args.mu_ic
    model_path = args.model_path
    time_frac = args.time_frac



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
    
    paths = glob(dataset_path +"/" + '*.npy')

    if (len(paths) == 0):
        raise Exception('No files found in dataset folder')
    
    # Load dataset
    ds_train, ds_valid = get_wave_datasets_timeslabs(paths, n_points=n_points, device=device, time_frac=time_frac)

    # Train and validation loaders
    train_dataloader =  torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    # Model
    if model_path:
        model = DeepONet(ds_train.us.shape[1], hidden_units, hidden_units)
        model.load_state_dict(torch.load(model_path))
    else:
        model = DeepONet(ds_train.us.shape[1], hidden_units, hidden_units)

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

    loss_col = make_loss_col_wave_eq(0.5)

    print(f'Training {str(model)} for {epochs} epochs')
    
    softadapt = SoftAdapt(2, beta=0.1)
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for (xt_batch, y_batch, u_batch) in train_dataloader:
            # update softadapt coefficients
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass through network
            pred = model(u_batch, xt_batch)

            loss_boundary = mu_boundary * loss_fn(pred, y_batch.view(-1))
            loss_collocation = mu_colloc * loss_col(model, u_batch, xt_batch)
            
            alphas = softadapt.get_alphas()

            loss = alphas[0] * loss_boundary + alphas[1] * loss_collocation
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            softadapt.update(torch.tensor([loss_boundary, loss_collocation]))
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            wandb.log({"train_loss_boundary": loss_boundary.item(), "epoch": epoch})
            wandb.log({"train_loss_collocation": loss_collocation.item(), "epoch": epoch})
            wandb.log({"alpha_boundary": alphas[0].item(), "epoch": epoch})
            wandb.log({"alpha_collocation": alphas[1].item(), "epoch": epoch})
                
        epoch_train_losses.append(np.mean(train_losses))
        wandb.log({"epoch_train_loss": epoch_train_losses[-1], "epoch": epoch})
        lr_scheduler.step()



        # Validation
        model.eval()
        validation_losses = []
        model.eval()
        for (xt_batch, y_batch, u_batch) in validation_dataloader:
            pred = model(u_batch, xt_batch)

            loss_boundary = mu_boundary * loss_fn(pred, y_batch.view(-1))
            loss_collocation = mu_colloc * loss_col(model, u_batch, xt_batch)
            
            alphas = softadapt.get_alphas()

            loss = alphas[0] * loss_boundary + alphas[1] * loss_collocation

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
    args.add_argument('--dataset', type=str, default='/work3/s216416/deeponet/data/2a/')
    args.add_argument('--model', type=str, default='FFNN')
    args.add_argument('--n_hidden', type=int, default=128)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=1e-3) 
    args.add_argument('--n_points', type=int, default=128)
    args.add_argument('--outputfolder', type=str, default='default')
    args.add_argument('--run_name', type=str, default='default')
    args.add_argument('--beta', type=float, default=0.1)
    args.add_argument('--mu_boundary', type=float, default=1.0)
    args.add_argument('--mu_colloc', type=float, default=0.0)
    args.add_argument('--mu_ic', type=float, default=1e-4)
    args.add_argument('--model_path', type=str, default=None)
    args.add_argument('--time_frac', type=float, default=1.0)

    args = args.parse_args()

    root = os.getcwd()

    model, train_losses, val_losses = train_model(args)

    output_folder = args.outputfolder
    
    print(f'Saving model and results to {output_folder}')

    save_model(model, root, output_folder)
    save_results(train_losses, root, output_folder, 'train_losses')
    save_results(val_losses, root, output_folder, 'val_losses')




