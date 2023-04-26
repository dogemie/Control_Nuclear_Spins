import numpy as np
from scipy.linalg import fractional_matrix_power
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


# theta = np.pi/4
# phi = np.pi/3

# print(degree(theta, phi))

def fidelity(rho1, rho2):
    hs_norm = np.linalg.norm(rho1 - rho2, ord='fro') # Calculate Frobenius norm of the difference between rho1 and rho2
    fid = np.trace(np.dot(rho1, rho2)) + hs_norm ** 2 / 2
    return fid

def fidelity2(rho1, rho2):
    """
    Calculate the fidelity between two density matrices using the Hilbert-Schmidt norm.

    Parameters:
        rho1 (numpy array): First density matrix.
        rho2 (numpy array): Second density matrix.

    Returns:
        Fidelity (float): The fidelity between the two density matrices.
    """
    sqrt_rho1 = np.sqrt(rho1)
    fidelity = np.real(np.trace(np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)))
    return fidelity


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        
        
# # Example usage
# rho1 = np.array([[0.20215035+0.j ,        0.28397675-0.28397675j], [0.28397675+0.28397675j, 0.79784965+0.j        ]])
# rho2 = np.array([[0.20551251-4.85353471e-18j, 0.28075492+2.90609348e-01j],[0.28075492-2.90609348e-01j, 0.79448749+4.85353471e-18j]])
# f = fidelity(rho1, rho2)
# print(f)

# f2 = state_fidelity(rho1, rho2)
# print(f2)


def angles_to_density_matrix(theta, phi):
    state_vector = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    density_matrix = np.outer(state_vector, np.conj(state_vector))
    return density_matrix

# Example usage
theta = 1.30081570812702
phi = 5.05600067687107
rho1 = degree(theta, phi)
print(rho1)
rho2 = np.array([[ 0.8519004 +4.25331715e-18j, -0.19932957+2.93996311e-01j],
 [-0.19932957-2.93996311e-01j,  0.1480996 -4.25331715e-18j]])

f = state_fidelity(rho1, rho2)
print(f)