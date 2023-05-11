import torch

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
        return 'DON_{}_{}_{}'.format(self.n_sensors, self.n_hidden, self.n_output)
