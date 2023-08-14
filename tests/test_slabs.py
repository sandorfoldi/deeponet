import torch
from src.slabs import SlabWeights


def test_slabweights_0():
    slab_weights = SlabWeights(
        num_x_slabs=10,
        num_t_slabs=10,
        x_min=0,
        x_max=1,
        t_max=1
    )

    assert slab_weights.num_weights == 100

def test_slabweights_1():
    slab_weights = SlabWeights(
        num_x_slabs=3,
        num_t_slabs=4,
        x_min=0,
        x_max=1,
        t_max=1
    )

    assert slab_weights.num_weights == 12

def test_slabweights_2():
    slab_weights = SlabWeights(
        num_x_slabs=10,
        num_t_slabs=10,
        x_min=0,
        x_max=1,
        t_max=1
    )

    assert slab_weights.x_to_idx(torch.tensor([0.1])) == torch.tensor([1])

def test_slabweights_3():
    slab_weights = SlabWeights(
        num_x_slabs=10,
        num_t_slabs=10,
        x_min=0,
        x_max=1,
        t_max=1
    )

    slab_weights.weights[0, 0] = 0.
    x_idx = slab_weights.x_to_idx(torch.tensor([0.05]))
    t_idx = slab_weights.t_to_idx(torch.tensor([0.05]))

    assert slab_weights(torch.tensor([[x_idx, t_idx]])) == 0.
