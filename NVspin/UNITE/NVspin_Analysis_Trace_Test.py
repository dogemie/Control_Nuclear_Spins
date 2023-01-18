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

def Rx(theta):
    return np.matrix([[cos(theta/2), -1j*sin(theta/2)],
                     [-1j*sin(theta/2), cos(theta/2)]])


def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2), 0],
                     [0, cos(phi/2)+1j*sin(phi/2)]])

def init():
    init = np.matrix([[1],[0]])
    return init
def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

idden = []
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

    #cost2 = ((float(np.abs(x_m-x_id)))**2+(float(np.abs(y_m-y_id)))**2+(float(np.abs(z_m-z_id)))**2)**(1/2)
    #print(rho_measure)

def problem2(deg):
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
    #cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    cost2 = ((np.abs(x_m-x_id)) * (np.abs(y_m-y_id)) * (np.abs(z_m-z_id)))
    return cost2


def problem3(deg):
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
    #cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    cost2 = (((np.abs(x_m-x_id))**2) * ((np.abs(y_m-y_id))**2) * ((np.abs(z_m-z_id)))**2)
    return cost2

def problem4(deg):
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
    #cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    cost2 = ((np.abs(x_m-x_id)) * (np.abs(y_m-y_id))) + ((np.abs(y_m-y_id)) * (np.abs(z_m-z_id))) + ((np.abs(z_m-z_id)) * (np.abs(x_m-x_id)))
    return cost2

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
    
def problem6(deg):
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
    #cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    cost2 = float(np.abs(x_m-x_id).real + np.abs(y_m-y_id).real + np.abs(z_m-z_id).real + -1j * (np.abs(x_m-x_id).imag + np.abs(y_m-y_id).imag + np.abs(z_m-z_id).imag))
    return cost2
    #
    #print(rho_measure)

bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수

def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz(phi)
    func = fz *fx
    mc = init()*init().T
    out = func*mc*func.getH()
    return out

date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
output1 = []
output2 = []
output3 = []
datapack = []
output4 = []

count = 5 #최적화 정도를 확인하기 위한 반복 작업 횟수
temp1 = np.zeros(count)
temp2 = np.zeros(count)
temp3 = np.zeros(count)
temp4 = np.zeros(count)
temp5 = np.zeros(count)
temp6 = np.zeros(count)
temp7 = np.zeros(count)
temp8 = np.zeros(count)
temp9 = np.zeros(count)
temp10 = np.zeros(count)
temp11 = np.zeros(count)
temp12 = np.zeros(count)

check = 0
check1 = 0
check2 = 0
check3 = 0
check4 = 0
check5 = 0
check6 = 0
check7 = 0
check8 = 0


ac1 = 0
ac2 = 0
ac3 = 0
ac4 = 0


for x in range(count):
    idden = rand_dm(2, density=1)
    # data = []
    # data = idden.data
    start = time.time()
    result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Powell")
    print(result1)
    end = time.time()
    final = end - start
    deft = degree(result1['x'][0], result1['x'][1]) #최적화된 값으로 density matrix를 구한다.

    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)

    output1.append([result1['x'], final, deft, car])
    temp1[x] = final
    temp5[x] = car
    start = time.time()
    result2 = scipy.optimize.minimize(problem2,deg,bounds=bounds,method="Powell")

    end = time.time()
    final = end - start
    deft = degree(result2['x'][0], result2['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + 
           (idden.data[0, 1] - deft[0, 1])**2 + 
           (idden.data[1, 1] - deft[1, 1])**2)**(1/2)

    output2.append([result2['x'], final, deft, car])
    temp2[x] = final
    temp6[x] = car
    start = time.time()
    result3 = scipy.optimize.minimize(problem3,deg,bounds=bounds,method="Powell") 
    end = time.time()
    final = end - start
    deft = degree(result3['x'][0], result3['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output3.append([result3['x'], final, deft, car])
    temp3[x] = final
    temp7[x] = car
    start = time.time()
    result4 = scipy.optimize.minimize(problem4,deg,bounds=bounds,method="Powell") 
    end = time.time()
    final = end - start
    deft = degree(result4['x'][0], result4['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output4.append([result4['x'], final, deft, car])
    temp4[x] = final
    temp8[x] = car
    #print(idden.data)
    datapack.append(idden.data) 
    
    start = time.time()
    result5 = scipy.optimize.minimize(problem5,deg,bounds=bounds,method="Powell")
    end = time.time()
    final = end - start
    deft = degree(result5['x'][0], result5['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    temp9[x] = car
    #print(idden.data)
    datapack.append(idden.data) 
    
    start = time.time()
    result6 = scipy.optimize.minimize(problem6,deg,bounds=bounds,method="Powell")
    end = time.time()
    final = end - start
    deft = degree(result6['x'][0], result6['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    temp10[x] = car
    #print(idden.data)
    datapack.append(idden.data) 
    
    start = time.time()
    result7 = scipy.optimize.differential_evolution(problem, bounds = bounds)
    end = time.time()
    final = end - start
    deft = degree(result7['x'][0], result7['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    temp11[x] = car
    
    start = time.time()
    result8 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Nelder-Mead")
    end = time.time()
    final = end - start
    deft = degree(result8['x'][0], result8['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    temp12[x] = car
    
    #print(idden.data)
    datapack.append(idden.data) 

    if(temp5[x] <= 0.1 or temp6[x] <= 0.1 or temp7[x] <= 0.1 or temp8[x] <= 0.1 or temp9[x] <= 0.1 or temp10[x] <= 0.1 or temp11[x] <= 0.1 or temp12[x] <= 0.1):
        check = check + 1
    if(temp5[x] <= 0.1):
        check1 = check1 + 1
        if(temp6[x] > 0.1 and temp7[x] > 0.1 and temp8[x] > 0.1 and temp9[x] > 0.1 and temp10[x] > 0.1 and temp11[x] > 0.1 and temp12[x] > 0.1):
            ac1 = ac1 + 1
    if(temp6[x] <= 0.1):
        check2 = check2 + 1
    if(temp7[x] <= 0.1):
        check3 = check3 + 1
    if(temp8[x] <= 0.1):
        check4 = check4 + 1
    if(temp9[x] <= 0.1):
        check5 = check5 + 1
    if(temp10[x] <= 0.1):
        check6 = check6 + 1
        if(temp5[x] > 0.1 and temp6[x] > 0.1 and temp7[x] > 0.1 and temp8[x] > 0.1 and temp9[x] > 0.1 and temp11[x] > 0.1 and temp12[x] > 0.1):
            ac2 = ac2 + 1
    if(temp11[x] <= 0.1):
        check7 = check7 + 1
        if(temp5[x]> 0.1 and temp6[x] > 0.1 and temp7[x] > 0.1 and temp8[x] > 0.1 and temp9[x] > 0.1 and temp10[x] > 0.1 and temp12[x] > 0.1):
            ac3 = ac3 + 1
    if(temp12[x] <= 0.1):
        check8 = check8 + 1
        if(temp5[x] > 0.1 and temp6[x] > 0.1 and temp7[x] > 0.1 and temp8[x] > 0.1 and temp9[x] > 0.1 and temp10[x] > 0.1 and temp11[x] > 0.1):
            ac4 = ac4 + 1
    print("Test Case" + str(x + 1) + " clear")

temp = np.arange(1, count + 1, 1)
print("Check : " + str(check) + " / " + str(count))
print("Check1 : " + str(check1) + " / " + str(count))
print("Check2 : " + str(check2) + " / " + str(count))
print("Check3 : " + str(check3) + " / " + str(count))
print("Check4 : " + str(check4) + " / " + str(count))
print("Check5 : " + str(check5) + " / " + str(count))
print("Check6 : " + str(check6) + " / " + str(count))
print("Check7 : " + str(check7) + " / " + str(count))
print("Check8 : " + str(check8) + " / " + str(count))

print("about Check1 : " + str(ac1) + " / " + str(count))
print("about Check6 : " + str(ac2) + " / " + str(count))
print("about Check7 : " + str(ac3) + " / " + str(count))
print("about Check8 : " + str(ac4) + " / " + str(count))

print(date)

fin1 = pd.DataFrame(output1)
fin1.rename(columns={0:'Nelder-Mead', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
#fin1.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Nelder_Mead_result_" + printdate + '.csv', index=false)
fin2 = pd.DataFrame(output2)
fin2.rename(columns={0:'Powell', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
#fin2.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Powell_result_" + printdate + '.csv', index=false)
fin3 = pd.DataFrame(output3)
fin3.rename(columns={0:'differential_evolution', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
#fin3.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/differential_evolution_result_" + printdate + '.csv', index=false)
fin4 = pd.DataFrame(output4)
fin4.rename(columns={0:'dual_annealing', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
#fin4.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/dual_annealing_result_" + printdate + '.csv', index=false)
pack = pd.DataFrame(datapack)
# fin4 = pd.DataFrame(output4)
pack.rename(columns={0:'Density Matrix'}, inplace=True)

fin = pd.concat([fin1, fin2, fin3, fin4, pack], axis=1)
fin.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Total_result_" + printdate + '.csv', index=false)

#plt.plot(temp, temp5, color='red', marker='o', label='Test1', linestyle='dashed', linewidth=2.5, markersize=12)
plt.plot(temp, temp6, color='blue', marker='o', label='Test2', linestyle='dashed', linewidth=2.5, markersize=12)
plt.plot(temp, temp7, color='green', marker='o', label='Test3', linestyle='dashed', linewidth=2, markersize=12)
plt.plot(temp, temp8, color='yellow', marker='o', label='Test4', linestyle='dashed', linewidth=2, markersize=12)
# plt.plot(temp, temp9, color='pink', marker='o', label='Test5', linestyle='dashed', linewidth=2, markersize=12)
# plt.plot(temp, temp10, color='orange', marker='o', label='Test6', linestyle='dashed', linewidth=2, markersize=12)
plt.title("Algorithm Error Comparison")
plt.xlabel("number of test case")
plt.ylabel("degree of error")
plt.legend()
plt.show()
