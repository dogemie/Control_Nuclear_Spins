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
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  
import math
import time
import matplotlib.pyplot as plt
start = time.time()
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
def Splus():
    Splus = np.array([[ 0, 1],
               [ 0, 0]])
    return Splus
def Sminus():
    Sminus = np.array([[ 0, 0],
               [ 1, 0]])
    return Sminus

def SD():
    SD = np.array([[ 1, 0],
               [ 0,-1j]])
    return SD

def SU():
    SU = np.array([[ 1, 0],
               [ 0,1j]])
    return SU

def ST():
    ST = np.array([[ 1, 1],
               [ 1,-1]])
    return ST

def SH():
    SH = np.array([[ 1, 1j],
               [ 1j, 1]])
    return SH

###2 Rotation operator(X-gate&Z-gate 2X2 matrices)

#X-gate는 Rx(theta)로 표현하고 Z-gate는 Rz(phi)로 표현합니다.
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
#혹시라도 에러가 나면 사용(허수계산이라 날수도 있음)

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])
'''                     
def Rz(phi):
    return np.matrix([[e**(-1j*phi/2),       0],
                     [0,                          e**(1j*phi/2)]])
'''

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

idden = []
###5 실행6

#problem(cost function)
def problem(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    p_m = np.trace(rho_measure*Splus())                         # Sigma Plus projection
    m_m = np.trace(rho_measure*Sminus())                        # Sigma Minus projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    p_id = np.trace(idden*Splus())                              # target state의 Sigma Plus projection
    m_id = np.trace(idden*Sminus())                             # target state의 Sigma Minus projection
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id) + np.abs(i_m - i_id) + np.abs(p_m - p_id) + np.abs(m_m - m_id)       # cost function
    return cost

def problem1(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    p_m = np.trace(rho_measure*Splus())                         # Sigma Plus projection
    m_m = np.trace(rho_measure*Sminus())                        # Sigma Minus projection
    d_m = np.trace(rho_measure*SD())                            # Sigma D projection
    u_m = np.trace(rho_measure*SU())                            # Sigma U projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    p_id = np.trace(idden*Splus())                              # target state의 Sigma Plus projection
    m_id = np.trace(idden*Sminus())                             # target state의 Sigma Minus projection
    d_id = np.trace(idden*SD())                                 # target state의 Sigma D projection
    u_id = np.trace(idden*SU())                                 # target state의 Sigma U projection
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id) + np.abs(i_m - i_id) + np.abs(p_m - p_id) + np.abs(m_m - m_id) + np.abs(d_m - d_id) + np.abs(u_m - u_id)       # cost function
    
    return cost

def problem2(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id) + np.abs(i_m - i_id)        # cost function
    return cost

def problem3(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    cost = ((np.abs(x_m-x_id))**2 + (np.abs(y_m-y_id))**2 + (np.abs(z_m-z_id))**2 + (np.abs(i_m - i_id))**2 )**(1/2)    # 실험값과 이론값의 비교 costfunction 반환
    #cost2 = ((float(np.abs(x_m-x_id)))**2+(float(np.abs(y_m-y_id)))**2+(float(np.abs(z_m-z_id)))**2)**(1/2)
    #print(rho_measure)
    plt.scatter(deg[1],deg[0])
    plt.pause(0.001)
    return cost

def problem4(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection

    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection

    
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id)      # cost function
    return cost

def problem5(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    p_m = np.trace(rho_measure*Splus())                         # Sigma Plus projection
    m_m = np.trace(rho_measure*Sminus())                        # Sigma Minus projection
    d_m = np.trace(rho_measure*SD())                            # Sigma D projection
    u_m = np.trace(rho_measure*SU())                            # Sigma U projection
    t_m = np.trace(rho_measure*ST())                            # Sigma T projection
    h_m = np.trace(rho_measure*SH())                            # Sigma H projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    p_id = np.trace(idden*Splus())                              # target state의 Sigma Plus projection
    m_id = np.trace(idden*Sminus())                             # target state의 Sigma Minus projection
    d_id = np.trace(idden*SD())                                 # target state의 Sigma D projection
    u_id = np.trace(idden*SU())                                 # target state의 Sigma U projection
    t_id = np.trace(idden*ST())                                 # target state의 Sigma T projection
    h_id = np.trace(idden*SH())                                 # target state의 Sigma H projection
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id) + np.abs(i_m - i_id) + np.abs(p_m - p_id) + np.abs(m_m - m_id) + np.abs(d_m - d_id) + np.abs(u_m - u_id) + np.abs(t_m - t_id) + np.abs(h_m - h_id)       # cost function
    return cost

def problem6(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    i_m = np.trace(rho_measure*I())                          # Identity projection
    d_m = np.trace(rho_measure*SD())                            # Sigma D projection
    u_m = np.trace(rho_measure*SU())                            # Sigma U projection
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    i_id = np.trace(idden*I())                               # target state의 Identity projection
    d_id = np.trace(idden*SD())                                 # target state의 Sigma D projection
    u_id = np.trace(idden*SU())                                 # target state의 Sigma U projection
    cost = np.abs(x_m - x_id) + np.abs(y_m - y_id) + np.abs(z_m - z_id) + np.abs(i_m - i_id) + np.abs(d_m - d_id) + np.abs(u_m - u_id)        # cost function
    return cost

def problem7(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection

    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection

    
    cost = ((np.abs(x_m - x_id))**2 + (np.abs(y_m - y_id))**2 + (np.abs(z_m - z_id))**2)**(1/2)      # cost function
    return cost

def problem8(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    i_m = np.trace(rho_measure*I())                          # Identity projection
    d_m = np.trace(rho_measure*SD())                            # Sigma D projection
    u_m = np.trace(rho_measure*SU())                            # Sigma U projection

    i_id = np.trace(idden*I())                               # target state의 Identity projection
    d_id = np.trace(idden*SD())                                 # target state의 Sigma D projection
    u_id = np.trace(idden*SU())                                 # target state의 Sigma U projection
    cost = np.abs(i_m - i_id) + np.abs(d_m - d_id) + np.abs(u_m - u_id)        # cost function
    return cost

def problem9(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    i_m = np.trace(rho_measure*I())                          # Identity projection
    d_m = np.trace(rho_measure*SD())                            # Sigma D projection
    u_m = np.trace(rho_measure*SU())                            # Sigma U projection

    i_id = np.trace(idden*I())                               # target state의 Identity projection
    d_id = np.trace(idden*SD())                                 # target state의 Sigma D projection
    u_id = np.trace(idden*SU())                                 # target state의 Sigma U projection
    cost = (np.abs(i_m - i_id))**2 + (np.abs(d_m - d_id))**2 + (np.abs(u_m - u_id))**2        # cost function
    return cost

def problem10(deg):  #Trash
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection

    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection

    
    cost = ((np.abs(x_m - x_id))*(np.abs(y_m - y_id))*(np.abs(z_m - z_id)))**(2)      # cost function
    return cost


bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수


#최적화 정도 측정
#idden과의 비교를 위하여 density matrix로 변환해주는 함수
def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz(phi)
    func = fz * fx
    mc = init()*init().T
    out = func*mc*func.getH()
    # print("func")
    # print(func)
    # print("func[0]")
    # print(func[0])
    # print("func[0][0]")
    # print(func[0][0])
    # print("func[0,0]")
    # print(func[0,0])
    # print("out")
    # print(out)
    # print("out[0]")
    # print(out[0])
    # print("out[0][0]")
    # print(out[0][0])
    
    return out

###6 결과 출력
#https://docs.scipy.org/doc/scipy/reference/optimize.html
###아래는 3가지의 옵티마이저를 사용하여 탐색 가능###

#output: 측정 값은 csv 파일에 저장하기 위한 리스트
#date: 현재 시간을 저장하기 위한 변수 파일 이름 지정에 사용
date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
output1 = []
output2 = []
output3 = []
datapack = []
output4 = []
#pip = [["Density Matrix", nan], [nan, "Matrix"]]
#최적화된 값의 변화가 없을 때 까지 작업을 반복한다.




fail = 0                                                    #최적화 실패 횟수
success = 0                                                 #최적화 성공 횟수

### x, y, z projection 저장

standard = 0.2                                              #최적화 정도의 기준 설정
min_stad = 1*e-10                                           #최적화 정도의 최소값 설정
count = 10
seccount = 15
vastand = 0.33



for x in range(count):                                         #반복 횟수 지정
    
    idden = rand_dm(2, density=1)
    ideal = []
    ideal = [np.trace(idden*Sx()),np.trace(idden*Sy()),np.trace(idden*Sz())] #target state의 x,y,z projection을 저장
    xx = 1e-8
    ff = 1e-8
    start = time.time()                                 #시간 측정 시작  
    for y in range(0, seccount):                                      #측정 횟수 지정 같은 작업을 여러번 진행할 경우를 대비하여 반복문 사용
        deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수
        
        result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft1 = degree(result1['x'][0], result1['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl1 = [np.trace(deft1*Sx()),np.trace(deft1*Sy()),np.trace(deft1*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car1 = ((idden.data[0, 0] - deft1[0, 0])**2 + (idden.data[0, 1] - deft1[0, 1])**2 + (idden.data[1, 1] - deft1[1, 1])**2)**(1/2) #최적화 정도 측정
        var1 = ((ideal[0] - deftl1[0])**2 + (ideal[1] - deftl1[1])**2 + (ideal[2] - deftl1[2])**2)**(1/2)
        
        theMin = 1
        var = 1
        
        result2 = scipy.optimize.minimize(problem1,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft2 = degree(result2['x'][0], result2['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl2 = [np.trace(deft2*Sx()),np.trace(deft2*Sy()),np.trace(deft2*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car2 = ((idden.data[0, 0] - deft2[0, 0])**2 + (idden.data[0, 1] - deft2[0, 1])**2 + (idden.data[1, 1] - deft2[1, 1])**2)**(1/2) #최적화 정도 측정
        var2 = ((ideal[0] - deftl2[0])**2 + (ideal[1] - deftl2[1])**2 + (ideal[2] - deftl2[2])**2)**(1/2)
        
        if(var2 < var):
            theMin = 2
            result = result2
            deft = deft2
            deftl = deftl2
            car = car2
            var = var2
        
        
        result3 = scipy.optimize.minimize(problem2,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft3 = degree(result3['x'][0], result3['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl3 = [np.trace(deft3*Sx()),np.trace(deft3*Sy()),np.trace(deft3*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car3 = ((idden.data[0, 0] - deft3[0, 0])**2 + (idden.data[0, 1] - deft3[0, 1])**2 + (idden.data[1, 1] - deft3[1, 1])**2)**(1/2) #최적화 정도 측정
        var3 = ((ideal[0] - deftl3[0])**2 + (ideal[1] - deftl3[1])**2 + (ideal[2] - deftl3[2])**2)**(1/2)
        
        if(var3 < var):
            theMin = 3
            result = result3
            deft = deft3
            deftl = deftl3
            car = car3
            var = var3
        
        result4 = scipy.optimize.minimize(problem3,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft4 = degree(result4['x'][0], result4['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl4 = [np.trace(deft4*Sx()),np.trace(deft4*Sy()),np.trace(deft4*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car4 = ((idden.data[0, 0] - deft4[0, 0])**2 + (idden.data[0, 1] - deft4[0, 1])**2 + (idden.data[1, 1] - deft4[1, 1])**2)**(1/2) #최적화 정도 측정
        var4 = ((ideal[0] - deftl4[0])**2 + (ideal[1] - deftl4[1])**2 + (ideal[2] - deftl4[2])**2)**(1/2)
        
        if(var4 < var):
            theMin = 4
            result = result4
            deft = deft4
            deftl = deftl4
            car = car4
            var = var4
        
        result5 = scipy.optimize.minimize(problem4,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft5 = degree(result5['x'][0], result5['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl5 = [np.trace(deft5*Sx()),np.trace(deft5*Sy()),np.trace(deft5*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car5 = ((idden.data[0, 0] - deft5[0, 0])**2 + (idden.data[0, 1] - deft5[0, 1])**2 + (idden.data[1, 1] - deft5[1, 1])**2)**(1/2) #최적화 정도 측정
        var5 = ((ideal[0] - deftl5[0])**2 + (ideal[1] - deftl5[1])**2 + (ideal[2] - deftl5[2])**2)**(1/2)
        
        if(var5 < var):
            theMin = 5
            result = result5
            deft = deft5
            deftl = deftl5
            car = car5
            var = var5
        
        result6 = scipy.optimize.minimize(problem5,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft6 = degree(result6['x'][0], result6['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl6 = [np.trace(deft6*Sx()),np.trace(deft6*Sy()),np.trace(deft6*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car6 = ((idden.data[0, 0] - deft6[0, 0])**2 + (idden.data[0, 1] - deft6[0, 1])**2 + (idden.data[1, 1] - deft6[1, 1])**2)**(1/2) #최적화 정도 측정
        var6 = ((ideal[0] - deftl6[0])**2 + (ideal[1] - deftl6[1])**2 + (ideal[2] - deftl6[2])**2)**(1/2)
        
        if(var6 < var):
            theMin = 6
            result = result6
            deft = deft6
            deftl = deftl6
            car = car6
            var = var6
       
        result7 = scipy.optimize.minimize(problem6,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        deft7 = degree(result7['x'][0], result7['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl7 = [np.trace(deft7*Sx()),np.trace(deft7*Sy()),np.trace(deft7*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car7 = ((idden.data[0, 0] - deft7[0, 0])**2 + (idden.data[0, 1] - deft7[0, 1])**2 + (idden.data[1, 1] - deft7[1, 1])**2)**(1/2) #최적화 정도 측정
        var7 = ((ideal[0] - deftl7[0])**2 + (ideal[1] - deftl7[1])**2 + (ideal[2] - deftl7[2])**2)**(1/2)
        
        if(var7 < var):
            theMin = 7
            result = result7
            deft = deft7
            deftl = deftl7
            car = car7
            var = var7
        
        # result8 = scipy.optimize.minimize(problem7,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        # deft8 = degree(result8['x'][0], result8['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl8 = [np.trace(deft8*Sx()),np.trace(deft8*Sy()),np.trace(deft8*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car8 = ((idden.data[0, 0] - deft8[0, 0])**2 + (idden.data[0, 1] - deft8[0, 1])**2 + (idden.data[1, 1] - deft8[1, 1])**2)**(1/2) #최적화 정도 측정
        # var8 = ((ideal[0] - deftl8[0])**2 + (ideal[1] - deftl8[1])**2 + (ideal[2] - deftl8[2])**2)**(1/2)
        
        # if(var8 < var):
        #     theMin = 8
        #     result = result8
        #     deft = deft8
        #     deftl = deftl8
        #     car = car8
        #     var = var8
            
        # result9 = scipy.optimize.minimize(problem8,deg,bounds=bounds,method="Powell", options = {'xtol' : xx, 'ftol' : ff })                        #Powell 최적화
        # deft9 = degree(result9['x'][0], result9['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl9 = [np.trace(deft9*Sx()),np.trace(deft9*Sy()),np.trace(deft9*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car9 = ((idden.data[0, 0] - deft9[0, 0])**2 + (idden.data[0, 1] - deft9[0, 1])**2 + (idden.data[1, 1] - deft9[1, 1])**2)**(1/2) #최적화 정도 측정
        # var9 = ((ideal[0] - deftl9[0])**2 + (ideal[1] - deftl9[1])**2 + (ideal[2] - deftl9[2])**2)**(1/2)
        
        # if(var9 < var):
        #     theMin = 9
        #     result = result9
        #     deft = deft9
        #     deftl = deftl9
        #     car = car9
        #     var = var9
        
        # if(var < vastand):
        #     end = time.time()
        #     final = end - start
        #     success = success + 1
        #     output1.append(["Case" + str(x + 1), "Method" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
        #     break
        
        # result10 = scipy.optimize.differential_evolution(problem3, bounds)                        #Powell 최적화
        # deft10 = degree(result10['x'][0], result10['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl10 = [np.trace(deft10*Sx()),np.trace(deft10*Sy()),np.trace(deft10*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car10 = ((idden.data[0, 0] - deft10[0, 0])**2 + (idden.data[0, 1] - deft10[0, 1])**2 + (idden.data[1, 1] - deft10[1, 1])**2)**(1/2) #최적화 정도 측정
        # var10 = ((ideal[0] - deftl10[0])**2 + (ideal[1] - deftl10[1])**2 + (ideal[2] - deftl10[2])**2)**(1/2)
        
        # if(var10 < var):
        #     theMin = 10
        #     result = result10
        #     deft = deft10
        #     deftl = deftl10
        #     car = car10
        #     var = var10
            
        # result11 = scipy.optimize.differential_evolution(problem4, bounds)                        #Powell 최적화
        # deft11 = degree(result11['x'][0], result11['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl11 = [np.trace(deft11*Sx()),np.trace(deft11*Sy()),np.trace(deft11*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car11 = ((idden.data[0, 0] - deft11[0, 0])**2 + (idden.data[0, 1] - deft11[0, 1])**2 + (idden.data[1, 1] - deft11[1, 1])**2)**(1/2) #최적화 정도 측정
        # var11 = ((ideal[0] - deftl11[0])**2 + (ideal[1] - deftl11[1])**2 + (ideal[2] - deftl11[2])**2)**(1/2)
        
        # if(var11 < var):
        #     theMin = 11
        #     result = result11
        #     deft = deft11
        #     deftl = deftl11
        #     car = car11
        #     var = var11
        
        result12 = scipy.optimize.minimize(problem3,deg,bounds=bounds,method="Nelder-Mead", options = {'xatol' : xx, 'fatol' : ff })                        #Powell 최적화
        deft12 = degree(result12['x'][0], result12['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl12 = [np.trace(deft12*Sx()),np.trace(deft12*Sy()),np.trace(deft12*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car12 = ((idden.data[0, 0] - deft12[0, 0])**2 + (idden.data[0, 1] - deft12[0, 1])**2 + (idden.data[1, 1] - deft12[1, 1])**2)**(1/2) #최적화 정도 측정
        var12 = ((ideal[0] - deftl12[0])**2 + (ideal[1] - deftl12[1])**2 + (ideal[2] - deftl12[2])**2)**(1/2)
        
        if(var12 < var):
            theMin = 12
            result = result12
            deft = deft12
            deftl = deftl12
            car = car12
            var = var12
        
        result13 = scipy.optimize.minimize(problem4,deg,bounds=bounds,method="Nelder-Mead", options = {'xatol' : xx, 'fatol' : ff })                        #Powell 최적화
        deft13 = degree(result13['x'][0], result13['x'][1])     #최적화된 값으로 density matrix를 생성
        deftl13 = [np.trace(deft13*Sx()),np.trace(deft13*Sy()),np.trace(deft13*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        car13 = ((idden.data[0, 0] - deft13[0, 0])**2 + (idden.data[0, 1] - deft13[0, 1])**2 + (idden.data[1, 1] - deft13[1, 1])**2)**(1/2) #최적화 정도 측정
        var13 = ((ideal[0] - deftl13[0])**2 + (ideal[1] - deftl13[1])**2 + (ideal[2] - deftl13[2])**2)**(1/2)
        
        if(var13 < var):
            theMin = 13
            result = result13
            deft = deft13
            deftl = deftl13
            car = car13
            var = var13
        
        if(var < vastand):
            end = time.time()
            final = end - start
            success = success + 1
            output1.append(["Case" + str(x + 1), "Method" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
            break
        
        # result14 = scipy.optimize.differential_evolution(problem3, bounds, strategy = 'best1exp')  
        # deft14 = degree(result14['x'][0], result14['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl14 = [np.trace(deft14*Sx()),np.trace(deft14*Sy()),np.trace(deft14*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car14 = ((idden.data[0, 0] - deft14[0, 0])**2 + (idden.data[0, 1] - deft14[0, 1])**2 + (idden.data[1, 1] - deft14[1, 1])**2)**(1/2) #최적화 정도 측정
        # var14 = ((ideal[0] - deftl14[0])**2 + (ideal[1] - deftl14[1])**2 + (ideal[2] - deftl14[2])**2)**(1/2)
        
        # if(var14 < var):
        #     theMin = 14
        #     result = result14
        #     deft = deft14
        #     deftl = deftl14
        #     car = car14
        #     var = var14
            
        # result15 = scipy.optimize.differential_evolution(problem4, bounds, strategy = 'best1exp')  
        # deft15 = degree(result15['x'][0], result15['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl15 = [np.trace(deft15*Sx()),np.trace(deft15*Sy()),np.trace(deft15*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car15 = ((idden.data[0, 0] - deft15[0, 0])**2 + (idden.data[0, 1] - deft15[0, 1])**2 + (idden.data[1, 1] - deft15[1, 1])**2)**(1/2) #최적화 정도 측정
        # var15 = ((ideal[0] - deftl15[0])**2 + (ideal[1] - deftl15[1])**2 + (ideal[2] - deftl15[2])**2)**(1/2)
        
        # if(var15 < var):
        #     theMin = 15
        #     result = result15
        #     deft = deft15
        #     deftl = deftl15
        #     car = car15
        #     var = var15
        
        # if(var < vastand):
        #     end = time.time()
        #     final = end - start
        #     success = success + 1
        #     output1.append(["Case" + str(x + 1), "Method" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
        #     break
        
        # result16 = scipy.optimize.differential_evolution(problem3, bounds, strategy = 'rand1exp')  
        # deft16 = degree(result16['x'][0], result16['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl16 = [np.trace(deft16*Sx()),np.trace(deft16*Sy()),np.trace(deft16*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car16 = ((idden.data[0, 0] - deft16[0, 0])**2 + (idden.data[0, 1] - deft16[0, 1])**2 + (idden.data[1, 1] - deft16[1, 1])**2)**(1/2) #최적화 정도 측정
        # var16 = ((ideal[0] - deftl16[0])**2 + (ideal[1] - deftl16[1])**2 + (ideal[2] - deftl16[2])**2)**(1/2)
        
        # if(var16 < var):
        #     theMin = 16
        #     result = result16
        #     deft = deft16
        #     deftl = deftl16
        #     car = car16
        #     var = var16
            
        # result17 = scipy.optimize.differential_evolution(problem4, bounds, strategy = 'rand1exp')  
        # deft17 = degree(result17['x'][0], result17['x'][1])     #최적화된 값으로 density matrix를 생성
        # deftl17 = [np.trace(deft17*Sx()),np.trace(deft17*Sy()),np.trace(deft17*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
        # car17 = ((idden.data[0, 0] - deft17[0, 0])**2 + (idden.data[0, 1] - deft17[0, 1])**2 + (idden.data[1, 1] - deft17[1, 1])**2)**(1/2) #최적화 정도 측정
        # var17 = ((ideal[0] - deftl17[0])**2 + (ideal[1] - deftl17[1])**2 + (ideal[2] - deftl17[2])**2)**(1/2)
        
        # if(var17 < var):
        #     theMin = 17
        #     result = result17
        #     deft = deft17
        #     deftl = deftl17
        #     car = car17
        #     var = var17
        
        
        # if(var < vastand):
        #     end = time.time()
        #     final = end - start
        #     success = success + 1
        #     output1.append(["Case" + str(x + 1), "Method" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
        #     break
        
        if(y == seccount - 1):
            end = time.time()
            final = end - start
            fail = fail + 1
            output1.append(["Case" + str(x + 1), "Fail" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
            break
        
    # end = time.time()
    # final = end - start
    # output1.append(["Case" + str(x + 1), "FAIL" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
    # fail = fail + 1
    # xx = xx/2
    # ff = ff/2
    # print("xx = " + str(xx))
    # print("var = " + str(var))
    # print("ff = " + str(ff))
    plt.show()
    print("Case" + str(x + 1) + " clear")                                               #측정이 끝난 경우 출력
        
print("Success : " + str(success) + "/" + str(count))                                                #측정 실패한 경우 출력
fin1 = pd.DataFrame(output1)
fin1.rename(columns={0:"Case", 1:"Used Algorithm", 2:'Theta, Phi', 3: 'time', 4: 'matrix', 5: "degree", 6: "Density Matrix", 7: "Projection", 8: "Projection"}, inplace=True)
fin1.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Result_" + printdate + '.csv', index=false)
print(date)                                                                             #측정이 끝난 시간 출력

#%%

# finalOutput = []
# finalOutput = output1 + output2 + output3
# output1[0][0] = "Nelder-Mead"
# output2[0][0] = "Powell"
# output3[0][0] = "COBYLA"
#print(idden)

#print(date)

# fin1 = pd.DataFrame(output1)
# fin1.rename(columns={0:'Nelder-Mead', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin1.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Nelder_Mead_result_" + printdate + '.csv', index=false)
# fin2 = pd.DataFrame(output2)
# fin2.rename(columns={0:'Powell', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin2.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Powell_result_" + printdate + '.csv', index=false)
# fin3 = pd.DataFrame(output3)
# fin3.rename(columns={0:'differential_evolution', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin3.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/differential_evolution_result_" + printdate + '.csv', index=false)
# fin4 = pd.DataFrame(output4)
# fin4.rename(columns={0:'dual_annealing', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin4.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/dual_annealing_result_" + printdate + '.csv', index=false)
# pack = pd.DataFrame(datapack)
# # fin4 = pd.DataFrame(output4)
# pack.rename(columns={0:'Density Matrix'}, inplace=True)

# fin = pd.concat([fin1, fin2, fin3, fin4, pack], axis=1)
# fin.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Total_result_" + printdate + '.csv', index=false)

#fin.to_csv('Powell_result_' + printdate + '.csv', index=false)

###7 결과 분석

#direc = 출발 지점에서의 방향
#fun = x 위치에서의 함수의 값 -> 최적화 정도라고 생각
#message = 메시지 문자열
#nfev = 목적 함수 호출 횟수
#njev = 자코비안 계산 횟수
#nit = x 이동 횟수
#status = 종료 상태, 0이면 최적화 성공
#x = 최적화 해
#xtol = x의 허용 오차
#ftol = func(xopt)에서 허용되는 상대 오류의 수
#https://www.desmos.com/scientific?lang=ko
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html


# %%
'''
result1 = scipy.optimize.differential_evolution(problem,bounds,atol=0.00001)
print("result1:")
print(result1) #결과 출력
print("/////////////////////////////////////////////////////////////////")

result11 = scipy.optimize.dual_annealing(problem,bounds)
print("result11:")
print(result11) #결과 출력
print("/////////////////////////////////////////////////////////////////")

result2 = scipy.optimize.dual_annealing(problem,bounds,maxiter=100)
print("result2:")
print(result2) #결과 출력
print("/////////////////////////////////////////////////////////////////")
'''

# %%
'''
result0 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.01,'ftol':0.01})
print("result0:")
print(result0) #결과 출력
output.append(result0)
print("/////////////////////////////////////////////////////////////////")

result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.001,'ftol':0.001})
print("result1:")
print(result1) #결과 출력
output.append(result1)
print("/////////////////////////////////////////////////////////////////")

result2 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.001,'ftol':0.001})
print("result2:")
print(result2) #결과 출력
output.append(result2)
print("/////////////////////////////////////////////////////////////////")

result3 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0001,'ftol':0.0001})
print("result3:")
print(result3) #결과 출력
output.append(result3)
print("/////////////////////////////////////////////////////////////////")

result4 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.00001,'ftol':0.00001})
print("result4:")
print(result4) #결과 출력
output.append(result4)
print("/////////////////////////////////////////////////////////////////")

result5 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.000001,'ftol':0.000001})
print("result5:")
print(result5) #결과 출력
output.append(result5)
print("/////////////////////////////////////////////////////////////////")

result6 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0000001,'ftol':0.0000001})
print("result6:")
print(result6) #결과 출력
output.append(result6)
print("/////////////////////////////////////////////////////////////////")

result7 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.00000001,'ftol':0.00000001})
print("result7:")
print(result7) #결과 출력
output.append(result7)
print("/////////////////////////////////////////////////////////////////")

result8 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.000000001,'ftol':0.000000001})
print("result8:")
print(result8) #결과 출력
output.append(result8)
print("/////////////////////////////////////////////////////////////////")
'''

#%%
'''
    idden = rand_dm(2, density=1)
    # data = []
    # data = idden.data
    start = time.time()
    result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Nelder-Mead")
    end = time.time()
    final = end - start
    deft = degree(result1['x'][0], result1['x'][1]) #최적화된 값으로 density matrix를 구한다.
    # print(idden[0][0])
    # print(deft[0][0])
    
    # deft1 = deft[0][0].tolist()
    # deft2 = deft[0][1].tolist()
    # deft3 = deft[1][0].tolist()
    # deft4 = deft[1][1].tolist()
    #find the distance between the target state(idden) and the result state(deft)
    # print(deft1)
    # print(deft2)
    # print(deft3)
    # print(deft4)
    # car = ((deft[0, 0])**2 + (deft[0, 1])**2 + (deft[1, 0])**2 + (deft[1, 1])**2)**(1/2)
    # car2 = ((idden[0][0])**2 + (idden[0][1])**2 + (idden[1][0])**2 + (idden[1][1])**2)**(1/2)
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    #print(car)
    output1.append([result1['x'], final, deft, car])
    
    start = time.time()
    result2 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Powell")
    end = time.time()
    final = end - start
    deft = degree(result2['x'][0], result2['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output2.append([result2['x'], final, deft, car])
    
    start = time.time()
    result3 = scipy.optimize.differential_evolution(problem,bounds=bounds)
    end = time.time()
    final = end - start
    deft = degree(result3['x'][0], result3['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output3.append([result3['x'], final, deft, car])
    
    start = time.time()
    result4 = scipy.optimize.dual_annealing(problem,bounds=bounds)
    end = time.time()
    final = end - start
    deft = degree(result4['x'][0], result4['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output4.append([result4['x'], final, deft, car])
    #print(idden.data)
    datapack.append(idden.data) 
    
    # result = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0001,'ftol':0.00001})  
    # output4.append([result['x'], result['fun']])
    
    print("test case" + str(x + 1) + " clear")
'''