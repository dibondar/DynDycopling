__doc__ = """
Python wrapper for C-library in integral_solver.c

Note: You must compile the C shared library
    gcc -O3 -fPIC -shared -o integral_solver.so integral_solver.c -lm -fopenmp -lnlopt
"""
import os
import ctypes
from ctypes import c_double, c_size_t, POINTER

# Load the shared library assuming that it is in the same directory
try:
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/integral_solver.so")
except OSError:
    raise NotImplementedError(
        """The library is absent. You must compile the C shared library using the following command:
            gcc -O3 -fPIC -shared -o integral_solver.so integral_solver.c -lm -fopenmp -lnlopt
        """
    )

############################################################################################
#
# return the sums I1 and I2
#
############################################################################################

lib.I1.argtypes = (POINTER(c_double), c_size_t)
lib.I1.restype = c_double


def I1(u):
    return lib.I1(u.ctypes.data_as(POINTER(c_double)), u.size)

lib.I2.argtypes = (POINTER(c_double), c_size_t)
lib.I2.restype = c_double


def I2(u):
    return lib.I2(u.ctypes.data_as(POINTER(c_double)), u.size)


def I1andI2(u):
    return I1(u), I2(u)

############################################################################################
#
#   double J(const double *u, const size_t u_size)
#
############################################################################################
lib.J.argtypes = (POINTER(c_double), c_size_t)
lib.J.restype = c_double


def J(u):
    return lib.J(u.ctypes.data_as(POINTER(c_double)), u.size)

############################################################################################
#
#   double minimizeJ(double *u, const size_t u_size)
#
############################################################################################

lib.minimizeJ.argtypes = (POINTER(c_double), c_size_t)
lib.minimizeJ.restype = c_double


def minimizeJ(u):
    return lib.minimizeJ(u.ctypes.data_as(POINTER(c_double)), u.size)
