import torch 


class SlabWeights:
    def __init__(self, num_x_slabs, num_t_slabs, x_min, x_max, t_max):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Slabweights using device: {self.device}')
        print('Warning, updating slab weights is not tested properly')
        self.num_x_slabs = num_x_slabs
        self.num_t_slabs = num_t_slabs
        self.weights = torch.ones([num_x_slabs, num_t_slabs], device=self.device)
        print(self.weights)
        self.x_min = torch.tensor(x_min)
        self.x_max = torch.tensor(x_max)
        self.t_max = torch.tensor(t_max)
        self.num_weights = num_x_slabs * num_t_slabs
    
    def x_to_idx(self, x):
        if x.min() < self.x_min or x.max() > self.x_max:
            raise ValueError(f'x must be in [{self.x_min}, {self.x_max}]')
        _ = ((x - self.x_min) / (self.x_max - self.x_min) * self.num_x_slabs)
        # cast to torch int tensor
        _ =  _.type(torch.int)
        _ = torch.clamp(_, min=0, max=self.num_x_slabs - 1)
        return _


    def t_to_idx(self, t):
        if t.min() < 0 or t.max() > self.t_max:
            raise ValueError(f't must be in [0, {self.t_max}]')
        _ = t / self.t_max * self.num_t_slabs
        # cast to torch int tensor
        _ = _.type(torch.int)    
        _ = torch.clamp(_, min=0, max=self.num_t_slabs - 1)
        return _
    
    def get_at_xt(self, xt):
        x, t = xt[:, 0], xt[:, 1]
        x_idxs = self.x_to_idx(x)
        t_idxs = self.t_to_idx(t)
        # print('----\n----\n')
        # print(self.weights)
        # print('----')
        # print(x_idxs)
        # print('----')
        # print(t_idxs)
        return self.weights[x_idxs, t_idxs]
    
    def set_at_xt(self, xt, val):
        x, t = xt[:, 0], xt[:, 1]
        x_idxs = self.x_to_idx(x)
        t_idxs = self.t_to_idx(t)
        self.weights[x_idxs, t_idxs] = val
    
    def update(self, vals):
        self.weights = vals.view([self.num_x_slabs, self.num_t_slabs])

    def tranform_loss(self, loss_batch, xt_batch):
        # batch -> slabs
        # loss_batch has shape [batch_size]
        # xt_batch has shape [batch_size, 2]

        avg_loss = torch.mean(loss_batch)
        x, t = xt_batch[:, 0], xt_batch[:, 1]
        x_idxs = self.x_to_idx(x)
        t_idxs = self.t_to_idx(t)
        losses = avg_loss * torch.ones([self.num_x_slabs, self.num_t_slabs], device=self.device)
        losses[x_idxs, t_idxs] = loss_batch

        return losses.view([-1])
    

if __name__ == '__main__':
    ws = SlabWeights(10, 10, 0, 1, 1)
    ws.weights[0, 0] = 0.
    ws.weights[0, 1] = 0.1
    ws.weights[0, 2] = 0.2
    ws.weights[0, 3] = 0.3


    xt = torch.tensor([[0.05, 0.05], [0.05, 0.15], [0.05, 0.25], [0.05, 0.35], [0.05, 0.45], [0.05, 0.55], [0.05, 0.65], [0.05, 0.75], [0.05, 0.85], [0.05, 0.95]])
    print(xt.shape)
    print(xt[:, 0])
    print(xt[:, 1])
    print(ws(xt))
        

