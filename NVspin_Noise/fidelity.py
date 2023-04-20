import numpy as np

def fidelity(rho1, rho2):
    sqrt_rho1 = np.sqrt(rho1)
    fidelity = np.trace(np.sqrt(np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)))
    return fidelity