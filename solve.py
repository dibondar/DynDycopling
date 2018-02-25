import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
#
#   Linking to the C code
#
#   Note: You must compile the C shared library
#       gcc -O3 -fPIC -shared -o integral_solver.so integral_solver.c -lm -fopenmp -lnlopt
#
##########################################################################################

from solver import I1andI2, J, minimizeJ

##########################################################################################
#
#   Start solving
#
##########################################################################################

t = np.linspace(0., 1, 100000 - 1)

u = np.cos(1. * 2. * np.pi * t)
u_original = u.copy()


print(I1andI2(u))

#J_python = np.sinh(u).sum() ** 2 + sum(np.sinh(u[k] - u[:(k+1)]).sum() for k in range(u.size)) ** 2

#J_c = J(u.ctypes.data_as(POINTER(c_double)), u.size)

#assert np.allclose(J_python, J_c)

minJ = minimizeJ(u)

print(I1andI2(u))

plt.subplot(121)
plt.title("function")
plt.plot(t, u_original, label='original guess')
plt.plot(t, u, '-', label='optimized')


plt.subplot(122)
plt.title("derivatives")
plt.plot(t, np.gradient(u_original), label='original guess')
plt.plot(t, np.gradient(u), '-', label='optimized')

plt.show()
