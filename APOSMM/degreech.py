import numpy as np

def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

print(todensity(0, 0.70710678))