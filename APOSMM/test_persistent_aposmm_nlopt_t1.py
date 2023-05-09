"""
Runs libEnsemble with APOSMM with the NLopt local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_nlopt.py
   python test_persistent_aposmm_nlopt_t1.py --nworkers 3 --comms local
   python test_persistent_aposmm_nlopt.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 3

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from math import gamma, pi, sqrt
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from time import time

# import numpy as np
from qutip import *
from sympy import *
from math import *
import scipy
from scipy import optimize
import random
import sympy as sp
from numpy import ndarray
import pandas as pd
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  
import math
# import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from scipy.linalg import fractional_matrix_power

# %%
###1 Pauli Matrices(2X2 matrices)               

def I():
    I  = np.array([[ 1, 0],
               [ 0, 1]])
    return I
def Sx():
    Sx = np.array([[ 0, 1],
               [ 1, 0]])
    return Sx
def Sy():
    Sy = np.array([[ 0,-1j],
               [1j, 0]])
    return Sy
def Sz():
    Sz = np.array([[ 1, 0],
               [ 0,-1]])
    return Sz

###2 Rotation operator(X-gate&Z-gate 2X2 matrices)

#X-gate는 Rx(theta)로 표현하고 Z-gate는 Rz(phi)로 표현합니다.
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])


###3 init & density matrix

#initial state |0> 으로 가정한 상태
# |0> = |vector> = |a> = |1> = |up> = |+> = |H> = |R> = |L>

def init():
    init = np.matrix([[1],[0]])
    return init

#state a|0> + b|1> 계수를 괄호안에 입력하여 density 구하는 함수
#density matrix는 2X2 matrix로 표현됩니다.
def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

# idden = []
# idden = rand_dm_ginibre(2, rank=1)

# idden = np.matrix([[0.5,0.5],[0.5,0.5]])
# idden = np.matrix([[1,0],[0,0]])
idden = [[1.00000000e+00-1.93983792e-17j, 5.87283099e-26+1.07906749e-09j],
 [5.87283099e-26-1.07906749e-09j, 1.16438664e-18-8.48838731e-36j]]

###4 실행

#problem(cost function)
#cost function은 target state와 계산값의 차이를 계산합니다.
# %%
# def problem(deg):
#     mc = init()*init().T                                        # |vector><vector|
#     gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
#     #rho_measure는 계산값(측정값)
#     rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
#     x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
#     y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
#     z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
#     i_m = np.trace(rho_measure*I())                          # Identity projection
#     #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
#     x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
#     y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
#     z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
#     i_id = np.trace(idden*I())                               # target state의 Identity projection
#     # 실험값과 이론값의 비교 costfunction 반환
#     cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id) + np.abs(i_m-i_id)
#     return cost
# %%

change_weight = 0.5


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)


# def makeNoise(array):
#     arral = [np.trace(array*Sx()), np.trace(array*Sy()), np.trace(array*Sz())]
#     # ns = (1 + random.uniform(-0.1, 0.1))
#     # np.random.seed(seed=100)
#     ns = np.random.poisson(lam=10, size=1)/10
#     # print(ns)
#     arre = np.zeros(3, dtype = 'complex_')
#     arre[0] = arral[0] * ns
#     sumarr = ((arre[0]) **2 + (arral[1]) **2 + (arral[2]) **2) ** (1/2)
#     arre[0] = arre[0] / sumarr
#     arre[1] = arral[1] / sumarr
#     arre[2] = arral[2] / sumarr
    
#     return arre


trace_time = [0, 5]
def problem(deg):
    mc = init()*init().T                                        # |vector><vector|
    timeErr = (deg[0] + deg[1]) * change_weight
    
    gates = np.inner(Rz(deg[1] + timeErr),Rx(deg[0]))                     # Universal Gate
    
    # timeCost = deg[2]                                           # timeCost는 시간에 따른 오차를 보정하기 위한 변수입니다.
    # timeCost = deg[1] + deg[2]
    # timeErr = timeCost * 0.1
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    # i_m = np.trace(rho_measure*I())                          # Identity projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    # x_id = noisy[0]
    # y_id = noisy[1]
    # z_id = noisy[2]
    # i_id = np.trace(idden*I())                               # target state의 Identity projection
    # 실험값과 이론값의 비교 costfunction 반환
    cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
    if(cost < trace_time[1]):
        trace_time[1] = cost
        trace_time[0] = timeErr
    return cost


# bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
# deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)]

def six_hump_camel_func(x):
    """ Six-Hump Camel function definition """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3

def six_hump_camel(H, persis_info, sim_specs, _):
    """Six-Hump Camel sim_f."""

    batch = len(H['x'])                            # Num evaluations each sim_f call.
    H_o = np.zeros(batch, dtype=sim_specs['out'])  # Define output array H

    for i, x in enumerate(H['x']):
        # H_o['f'][i] = three_hump_camel_func(x)     # Function evaluations placed into H
        H_o['f'][i] = six_hump_camel_func(x)     # Function evaluations placed into H

    return H_o, persis_info





# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    if is_manager:
        start_time = time()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 50,
            "sample_points": np.round(minima, 1),
            "localopt_method": "LN_BOBYQA",
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "xtol_abs": 1e-6,
            "ftol_abs": 1e-6,
            "dist_to_bound_multiple": 0.5,
            "max_active_runs": 6,
            "lb": np.array([0, 0]),
            "ub": np.array([pi, 2 * pi]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 1000}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        print("[Manager]: Time taken =", time() - start_time, flush=True)
 
        tol = 1e-5
        for m in minima:
            # The minima are known on this test problem.
            # We use their values to test APOSMM has identified all minima
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            # assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol

        save_libE_output(H, persis_info, __file__, nworkers)