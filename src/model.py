import torch
from wave_loader import get_wave_datasets

class DeepONet1DCNN(torch.nn.Module):
    """
        "both NNs are general architectures, e.g., the branch NN can be replaced with a CNN or a ResNet."
        - https://www.sciencedirect.com/science/article/abs/pii/S0045782522001207 

        Idea is simply to reshape the input of the branch net to be a 2D image and use that
        for prediections.
    
    """
    
    def __init__(self, n_sensors, n_hidden, n_output):
        super(DeepONet1DCNN, self).__init__()
        self.n_sensors = n_sensors
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Branch net
        """
        self.B1_conv = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.B1_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B2_conv = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.B2_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B3_conv = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.B3_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B4 = torch.nn.Linear(32, n_output)
        """
        # one dimensional convolutions
        self.B1_conv = torch.nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
        self.B1_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.B2_conv = torch.nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.B2_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.B3_conv = torch.nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.B3_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.B4 = torch.nn.Linear(384, n_output)

        # Trunk net
        self.T1 = torch.nn.Linear(2, n_hidden)
        self.T2 = torch.nn.Linear(n_hidden, n_hidden)
        self.T3 = torch.nn.Linear(n_hidden, n_hidden)
        self.T4 = torch.nn.Linear(n_hidden, n_hidden)
        self.T5 = torch.nn.Linear(n_hidden, n_output)

        self.activation = torch.sin

    def forward(self, u, y):
        # Pass through branch net
        # b = self.activation(self.B1_pool(self.B1_conv(u.view(-1, 1, 10, 10))))
        b = self.activation(self.B1_pool(self.B1_conv(u.unsqueeze(1))))
        b = self.activation(self.B2_pool(self.B2_conv(b)))
        b = self.activation(self.B3_pool(self.B3_conv(b)))
        b = self.B4(b.flatten(start_dim=1))
        # b = self.B4(b)


        # Pass through trunk net
        t = self.activation(self.T1(y))
        t = self.activation(self.T2(t))
        t = self.activation(self.T3(t))
        t = self.activation(self.T4(t))
        t = self.T5(t)

        # Combine
        output = torch.einsum('ij,ij->i', b, t)

        return output

    def __repr__(self):
        return 'DON_CNN1D_{}_{}_{}'.format(self.n_sensors, self.n_hidden, self.n_output)

class DeepONet2DCNN(torch.nn.Module):
    """
        "both NNs are general architectures, e.g., the branch NN can be replaced with a CNN or a ResNet."
        - https://www.sciencedirect.com/science/article/abs/pii/S0045782522001207 

        Idea is simply to reshape the input of the branch net to be a 2D image and use that
        for prediections.
    
    """
    
    def __init__(self, n_sensors, n_hidden, n_output):
        super(DeepONet2DCNN, self).__init__()
        self.n_sensors = n_sensors
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Branch net
        self.B1_conv = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.B1_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B2_conv = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.B2_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B3_conv = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.B3_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.B4 = torch.nn.Linear(32, n_output)

        # Trunk net
        self.T1 = torch.nn.Linear(2, n_hidden)
        self.T2 = torch.nn.Linear(n_hidden, n_hidden)
        self.T3 = torch.nn.Linear(n_hidden, n_hidden)
        self.T4 = torch.nn.Linear(n_hidden, n_hidden)
        self.T5 = torch.nn.Linear(n_hidden, n_output)

        self.activation = torch.sin

    def forward(self, u, y):
        # Pass through branch net
        b = self.activation(self.B1_pool(self.B1_conv(u.view(-1, 1, 10, 10))))
        b = self.activation(self.B2_pool(self.B2_conv(b)))
        b = self.activation(self.B3_pool(self.B3_conv(b)))
        b = self.B4(b.view(-1, 32))

        # Pass through trunk net
        t = self.activation(self.T1(y))
        t = self.activation(self.T2(t))
        t = self.activation(self.T3(t))
        t = self.activation(self.T4(t))
        t = self.T5(t)

        # Combine
        output = torch.einsum('ij,ij->i', b, t)

        return output

    def __repr__(self):
        return 'DON_CNN2D_{}_{}_{}'.format(self.n_sensors, self.n_hidden, self.n_output)

# Neural network 
class DeepONet(torch.nn.Module):
    def __init__(self, n_sensors, n_hidden, n_output):
        super(DeepONet, self).__init__()
        # Parameters
        self.n_sensors = n_sensors
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Branch net
        self.B1 = torch.nn.Linear(n_sensors, n_hidden)
        self.B2 = torch.nn.Linear(n_hidden, n_hidden)
        self.B3 = torch.nn.Linear(n_hidden, n_hidden)
        self.B4 = torch.nn.Linear(n_hidden, n_hidden)
        self.B5 = torch.nn.Linear(n_hidden, n_output)

        # Batch norm
        self.BN1 = torch.nn.BatchNorm1d(n_hidden)
        self.BN2 = torch.nn.BatchNorm1d(n_hidden)
        self.BN3 = torch.nn.BatchNorm1d(n_hidden)
        self.BN4 = torch.nn.BatchNorm1d(n_hidden)

        # Trunk net
        self.T1 = torch.nn.Linear(2, n_hidden)
        self.T2 = torch.nn.Linear(n_hidden, n_hidden)
        self.T3 = torch.nn.Linear(n_hidden, n_hidden)
        self.T4 = torch.nn.Linear(n_hidden, n_hidden)
        self.T5 = torch.nn.Linear(n_hidden, n_output)

        # batch norm
        self.TBN1 = torch.nn.BatchNorm1d(n_hidden)
        self.TBN2 = torch.nn.BatchNorm1d(n_hidden)
        self.TBN3 = torch.nn.BatchNorm1d(n_hidden)
        self.TBN4 = torch.nn.BatchNorm1d(n_hidden)

        self.activation = torch.sin

    def set_batch(self, batch):
        self.batch = batch

    def forward(self, u, y):
        # Pass through branch net
        b = self.activation(self.BN1(self.B1(u)))
        b = self.activation(self.BN2(self.B2(b)))
        b = self.activation(self.BN3(self.B3(b)))
        b = self.activation(self.BN4(self.B4(b)))
        b = self.B5(b)

        # Pass through trunk net
        t = self.activation(self.TBN1(self.T1(y)))
        t = self.activation(self.TBN2(self.T2(t)))
        t = self.activation(self.TBN3(self.T3(t)))
        t = self.activation(self.TBN4(self.T4(t)))
        t = self.T5(t)

        # Combine
        output = torch.einsum('ij,ij->i', b, t)

        return output
    
    def __repr__(self):
        return 'DON_Dense_{}_{}_{}'.format(self.n_sensors, self.n_hidden, self.n_output)