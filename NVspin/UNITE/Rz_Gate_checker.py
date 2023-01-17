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
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  
import math
import time
# start = time.time()
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

# def Rz1(phi):
#     return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
#                      [0,                          cos(phi/2)+1j*sin(phi/2)]])
                   
# def Rz2(phi):
#     return np.matrix([[e**(-1j*phi/2),       0],
#                      [0,                          e**(1j*phi/2)]])


# a = float(input("Enter the phi value: "))
# print("phi = ", a)
# print("Rz1(phi) = ", Rz1(a))
# print("Rz2(phi) = ", Rz2(a))
# # if(Rz1(a) == Rz2(a)):
# #     print("Rz1(phi) = Rz2(phi)")
# # else:
# #     print("Rz1(phi) != Rz2(phi)")

# #X-gate는 Rx(theta)로 표현하고 Z-gate는 Rz(phi)로 표현합니다.
# def Rx(theta):
#     return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
#                     [-1j*sin(theta/2),     cos(theta/2)]])

# b = float(input("Enter the theta value: "))
# print("theta = ", b)
# print("Rx(theta) = ", Rx(b))


# print("Rz1(phi) * Rx(theta) = ", Rz1(a)*Rx(b))

def init():
    init = np.matrix([[1],[0]])
    return init

idden = []
# mc = init()*init().T
# print(mc)

def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
#혹시라도 에러가 나면 사용(허수계산이라 날수도 있음)

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])
    
bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수

def problem5(deg):
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
    x_j = (np.abs(x_m-x_id).imag)**2
    y_j = (np.abs(y_m-y_id).imag)**2
    z_j = (np.abs(z_m-z_id).imag)**2
    print("x_j : " + str(x_j))
    print("y_j : " + str(y_j))
    print("z_j : " + str(z_j))
    if(x_j < 0):
        x_j = x_j * -1
    if(y_j < 0):
        y_j = y_j * -1
    if(z_j < 0):
        z_j = z_j * -1
    im = x_j + y_j + z_j
    cost = ((np.abs(x_m-x_id).real)**2 + (np.abs(y_m-y_id).real)**2 + (np.abs(z_m-z_id).real)**2)**(1/2) + im**(1/2)   # 실험값과 이론값의 비교 costfunction 반환
    return cost
    #
    #print(rho_measure)
    


for x in range(20):
    idden = rand_dm(2, density=1)
    temp9 = []
    start = time.time()
    result5 = scipy.optimize.minimize(problem5,deg,bounds=bounds,method="Powell")
    end = time.time()
    final = end - start
    deft = degree(result5['x'][0], result5['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    temp9[x] = car
    #print(idden.data)