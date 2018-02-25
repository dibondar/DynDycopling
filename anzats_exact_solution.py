import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from solver import I1, I1andI2


t = np.linspace(0., 1, 10000 - 1)
u = np.cos(1. * 2. * np.pi * t)

print(I1andI2(u))

def F(x):
    u[int(u.size / 2)] = x
    return I1(u)

u[int(u.size / 2)] = brentq(F, -1.5902, -1.5901, xtol=1e-14)

print(I1andI2(u))

plt.title("Numerically exact solution")
plt.subplot(121)
plt.title("function")
plt.plot(t, u, '-')


plt.subplot(122)
plt.title("derivatives")
plt.plot(t, np.gradient(u), '-')

plt.show()