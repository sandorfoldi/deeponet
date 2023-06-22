import numpy as np
import matplotlib.pyplot as plt


def ic_gm(aa, ii):
    return lambda x: sum([np.exp(-a * (x - i) ** 2) for a in aa for i in ii])

x = np.linspace(0, 10, 1000)
fig, axes = plt.subplots(4, 1)
print(axes.shape)
n = np.random.randint(0, 100)
m = [np.random.random()*10 for _ in range(n)]
v = [np.random.random()*10 for _ in range(n)]
ic = ic_gm(m, v)
axes[0].plot(x, ic(x), label="orig")
plt.tight_layout()
comps = 1000
fu = np.fft.fft(ic(x))
"""
fu = np.array([[i.real, i.imag] for i in fu])
# fu = fu.reshape(-1)
# print(fu.shape)
fu0 = fu[:n+1]
fu1 = fu[-n-1:]
fu1 = np.flip(fu1, axis=0)
fu_diff = fu0 - fu1
"""
fu_ = np.array([[i.real, i.imag] for i in fu])
print(fu_.shape)
axes[1].plot(fu_, label="fft")
axes[1].set_ylim(-20, 20)
fu[int(comps/4):-int(comps/4)] = 0
rec = np.fft.ifft(fu, n=1000)
axes[2].plot(x, rec, label="rec")
axes[3].plot(x, ic(x)-rec)

plt.legend()
plt.show()
