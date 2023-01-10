# %%
# NV Single Qubit에서 원하는 target state로 가기위한 theta와 phi를 계산하는 알고리즘

# 본코드는 VSCODE 에디터를 사용하여 Jupyter확장으로 작성하였습니다 By bin 2022.08.31
# 모든 변수들은 함수선언(def)으로 되어있는데 이는 여러가지 알고리즘을 실험하기 위함입니다.
# 아래의 주석처리에 따라 target state를 superposition과 random한 state중 한가지를 고를 수 있습니다.

import numpy as np
from qutip import *
from sympy import *
from math import *
import scipy
from scipy import optimize
import random
import sympy as sp
from numpy import ndarray
import pandas as pd

#Pauli Matrices(2X2 matrices)
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

#Rotation operator(X-gate&Z-gate 2X2 matrices)
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
'''혹시라도 에러가 나면 사용(허수계산이라 날수도 있음)
def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])
'''                     
def Rz(phi):
    return np.matrix([[e**(-1j*phi/2),       0],
                     [0,                          e**(1j*phi/2)]])

#initial state |0> 으로 가정한 상태
def init():
    init = np.matrix([[1],[0]])
    return init

#state a|0> + b|1> 계수를 괄호안에 입력하여 density 구하는 함수
def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

#ideal한 target state(중첩상태를 final로 둘 경우)
ideal = np.matrix([[1/sqrt(2)],[1/sqrt(2)]])

#위의 target state의 density matrix
idden = np.matrix([[1/2,1/2],
                    [1/2,1/2]])

'''
#랜덤한 target state를 생성하여 density matrix까지 구하는 함수 
while 1 :
        U=rand_unitary(2) #qutip 라이브러리에 있는 랜덤한 qubit state를 생성해주는 함수
        a=U[0,0]
        b=U[1,0]
        UU=np.array([[a],[b]])
        TU = UU.T
        CU = TU.conjugate()
        idden = UU*CU
        if np.abs(AU[1,1]) != 0:
            break
'''


#problem(cost function)
def problem(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    return cost

bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수

###아래는 3가지의 옵티마이저를 사용하여 탐색 가능###
#result = scipy.optimize.differential_evolution(problem,bounds,atol=0.00001)
#result = scipy.optimize.dual_annealing(problem,bounds,maxiter=100)
result = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.00001,'ftol':0.00001})  

print(result) #결과 출력

# %%
