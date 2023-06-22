import numpy as np
import matplotlib.pyplot as plt


def ic_gm_(aa, ii):
    return lambda x: sum([np.exp(-a * (x - i) ** 2) for a in aa for i in ii])

def ic_gm(mm, vv):
    return lambda x: sum([1/np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v)) for m in mm for v in vv])

if __name__ == '__main__':
    num_g = 10
    m_min = 0
    m_max = 10
    v_min = 10
    v_max = 10
    x0 = 0
    x1 = 10
    xn = 1000
    x = np.linspace(x0, x1, xn)
    fig, axes = plt.subplots(4, 4)
    for i in range(16):
        # n = np.random.randint(0, 100)
        m = [np.random.random()*(m_max-m_min) + m_min for _ in range(num_g)]
        v = [np.random.random()*(v_max-v_min) + v_min for _ in range(num_g)]
        ic = ic_gm(m, v)
        f = ic_gm([1., 5., 10., 20.], [1., 2., 3., 4.])
        axes[i//4, i%4].plot(x, ic(x))
    plt.tight_layout()
    plt.show()