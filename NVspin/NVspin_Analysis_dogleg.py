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

'''
###4 density matrix를 이용한 cost function

#ideal한 target state(중첩상태를 final로 둘 경우)
#Not Used
ideal = np.matrix([[1/sqrt(2)],[1/sqrt(2)]])
'''
#위의 target state의 density matrix
# idden = np.matrix([[3/8,1/5],
#                     [1/5,5/8]])



#랜덤한 target state를 생성하여 density matrix까지 구하는 함수 
# while 1 :
#         U=rand_unitary(2) #qutip 라이브러리에 있는 랜덤한 qubit state를 생성해주는 함수
#         a=U[0,0]
#         b=U[1,0]
#         UU=np.array([[a],[b]])
#         TU = UU.T                                              # Transpose, 전치 배열
#         CU = TU.conjugate()                                    # conjugate = conj <= 복소수?
#         idden = UU*CU
#         if np.abs(idden[1,1]) != 0:
#             break


#qutip 라이브러리에 있는 랜덤한 density matrix를 생성해주는 함수
#idden = rand_dm(2, density=1)



#matrix 지정
idden = np.matrix([[0.49168943+0.j,-0.09028809+0.0045274j],
                    [-0.09028809+0.0045274j,0.50831057+0.j]])


###5 실행

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
    #cost2 = ((float(np.abs(x_m-x_id)))**2+(float(np.abs(y_m-y_id)))**2+(float(np.abs(z_m-z_id)))**2)**(1/2)
    return cost

bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수


###6 결과 출력
#https://docs.scipy.org/doc/scipy/reference/optimize.html
###아래는 3가지의 옵티마이저를 사용하여 탐색 가능###

output = []

# xx = 0.1
# ff = 0.1
# for x in range(1, 60): #최적화 정도를 확인하기 위한 반복 작업
#     result = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol': xx,'ftol': ff})
#     output.append(result)
#     xx = xx * 0.3
#     ff = ff * 0.3
#     print("result" + str(x) + " clear") 


#최적화된 값의 변화가 없을 때 까지 작업을 반복한다.
#xx = 1
ff = 1
count = 1
bbb = 100                                   #이전의 cost function값을 저장하기 위한 변수
ccc = 100                                   #이전의 이전의 cost function값을 저장하기 위한 변수
while true:
    result = scipy.optimize.minimize(problem,deg,bounds=bounds,method="dogleg", options={'gtol': ff})
    output.append(result)
    aaa = output[count - 1]['fun']          #Temp Memory to compate output 'fun' value
    if(float(aaa) >= float(bbb) and float(aaa) >= float(ccc)):
        print("minimize clear")
        break
    if(count >= 100):
        print("minimize fail")
        break
    ccc = bbb
    bbb = aaa
    if(count == 1):
        #xx = xx * 0.5
        ff = ff * 0.5
        print("result" + str(count) + " clear") 
        count = count + 1
    else:
        #xx = xx * 0.25
        ff = ff * 0.25
        print("result" + str(count) + " clear")
        count = count + 1
    



fin = pd.DataFrame(output)
print(idden)
date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)
fin.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/dogleg/dogleg_result_" + printdate + '.csv', index=false)
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

end = time.time()
final = end - start
print("running time: " + str(final))

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