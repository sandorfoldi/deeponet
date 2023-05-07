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

        # Trunk net
        self.T1 = torch.nn.Linear(2, n_hidden)
        self.T2 = torch.nn.Linear(n_hidden, n_hidden)
        self.T3 = torch.nn.Linear(n_hidden, n_output)
        self.T4 = torch.nn.Linear(n_hidden, n_output)
        self.T5 = torch.nn.Linear(n_hidden, n_output)

        self.activation = torch.sin

        self.batch = True # Flag for batch training

    def set_batch(self, batch):
        self.batch = batch

    def forward(self, u, y):
        # Pass through branch net
        b = self.activation(self.B1(u))
        b = self.activation(self.B2(b))
        b = self.activation(self.B3(b))
        b = self.activation(self.B4(b))
        b = self.B5(b)

        # Pass through trunk net
        t = self.activation(self.T1(y))
        t = self.activation(self.T2(t))
        t = self.activation(self.T3(t))
        t = self.activation(self.T4(t))
        t = self.T5(t)

        # Combine
        if self.batch:
            output = torch.einsum('ij,ij->i', b, t)
        else:
            output = b @ t

        return output
    
    def __repr__(self):
        return 'DON_{}_{}_{}'.format(self.n_sensors, self.n_hidden, self.n_output)
