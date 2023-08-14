# test with pytest

# import pytest
import torch
import numpy as np
from src.softadapt import SoftAdapt

def test_softadapt_0():
    softadapt = SoftAdapt(2, beta=0.1)
    assert np.allclose(softadapt.get_alphas(), np.array([1., 1.]))


def test_softadapt_1():
    softadapt = SoftAdapt(2, beta=0.1)
    softadapt.update(torch.tensor([10., 1.]))
    softadapt.update(torch.tensor([10., 1.]))
    assert np.allclose(softadapt.get_alphas(), np.array([.5, .5]))


def test_softadapt_2():
    softadapt = SoftAdapt(4, beta=0.1)
    softadapt.update(torch.tensor([10., 10., 10., 10.]))
    softadapt.update(torch.tensor([10., 10., 10., 10.]))
    assert np.allclose(softadapt.get_alphas(), np.array([0.25, 0.25, 0.25, 0.25]))


def test_softadapt_3():
    softadapt = SoftAdapt(4, beta=0.1)
    softadapt.update(torch.tensor([0., 0., 0., 0.]))
    softadapt.update(torch.tensor([1000., 0., 0., 0.]))
    assert np.allclose(softadapt.get_alphas(), np.array([1., 0., 0., 0.]))


def test_softadapt_4():
    softadapt = SoftAdapt(4, beta=-0.1)
    softadapt.update(torch.tensor([0., 0., 0., 0.]))
    softadapt.update(torch.tensor([-100., 100., 100., 100.]))
    print(softadapt.get_alphas())
    assert np.allclose(softadapt.get_alphas(), np.array([1., 0., 0., 0.]))

