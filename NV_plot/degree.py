import numpy as np
from qutip import *
from sympy import *
from math import *

def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

def init():
    init = np.matrix([[1],[0]])
    return init

def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz(phi)
    func = fz * fx
    mc = init()*init().T
    out = func*mc*func.getH()

    return out


theta = np.pi/4
phi = np.pi/3

print(degree(theta, phi))