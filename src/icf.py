import numpy as np


def polynomial(point):
    return lambda x: sum([x**i * point[i] for i in range(len(point))])

def fourier(point):
    return lambda x: sum([point[i] * np.sin(i * x) for i in range(len(point))])