#%%
#본 코드는 initial state -> target state로 만들기 위해 13C를 컨트롤 하기 위한 N,tau를 구하는 알고리즘입니다.
from toqito.channels import partial_trace
from qutip import *
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
import math
import matplotlib.pyplot as plt
from scipy import optimize
import random
from math import *
import pandas as pd

def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]
    return UA

i   = 1j #1j
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/i*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
#sz  = [1, 0, 0; 0, -1, 0; 0, 0, 0]
I   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -i],[0, 0, 0], [i, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])
# Rotation matrix projected into 2 level system
Sxp  = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sxm  = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Syp  = 1/i*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
Sym  = 1/i*np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

# Pauli basis for 13C nuclear spin
Ix  = 1/2*np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/2/i*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
 

# Define sweep parameters
Sweep = 1001
N = Sweep
B = 403 #[G] magnetic field

    

# 13C nuclear spin parameters
gammaN = 2*pi*1.071e-3 #[MHz/G]

Al    = 2*math.pi*0.1 #[MHz] # A_|| hyperfine term
Ap = 2*pi*0.1 #[MHz] # A_per hyperfine term

T = 5; # sweep tau [us]
t = np.linspace(0,T,N)
n = 32; # number of pi pulses

## Define gate operations
# Single Q ms=+1
U090xp = UO(1,0,pi/4,0,0)
U090xmp = UO(1,0,-pi/4,0,0)
U090yp = UO(1,0,pi/4,pi/2,0)
U090ymp = UO(1,0,-pi/4,pi/2,0)
U180xp = UO(1,0,pi/2,0,0)
U180xmp = UO(1,0,-pi/2,0,0)

#Single Q ms=-1
U090xm = UO(0,1,pi/4,0,0)
U090xmm = UO(0,1,-pi/4,0,0)
U180xm = UO(0,1,pi/2,0,0)
U180xmm = UO(0,1,pi/2,0,0)

# Define initial state of the system (스핀상태)

irho_p = np.array([[1,0,0],[0,0,0],[0,0,0]]) #;0,0,0;0,0,0]

irho_m = np.array([[0,0,0],[0,0,0],[0,0,1]]) #0,0,0;0,0,1]

irho_z = np.array([[0,0,0],[0,1,0],[0,0,0]]) #0,1,0;0,0,0]

irho_mix = np.array([[1/2,0,0],[0,1/2,0],[0,0,0]])

irho_Z = np.array([[0,0,0],[0,0,0],[0,0,1]])

irho_MIX = np.array([[1/2,0,0],[0,0,0],[0,0,1/2]])

irho = np.kron(irho_z,irho_p)

#Initialization
rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) # superposition state on NV

#resornant tau를 찾는 코드
Sa= []


for h in range(N):
    ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz)
    
    eigvals = np.linalg.eigh(ham)[0]                                      
    eigvecs = -1*np.linalg.eigh(ham)[1]
    E = np.diag(eigvals)                                                  
    U_H= eigvecs.conj().T                                                  

    
    #free evolution unitary operator
    U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*t[h]/2)@U_H)                                        # for tau/2
    U_e  = (U_H.conj().T)@(linalg.expm(-i*E*t[h])@U_H)                                          # for tau
    rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)                                                      # first tau/2
    for k in range(1,n):
        rho_1 = U_e @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N과 tau를 N개 생성
        
    rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T)   # N & tau
    rho_3 = np.kron(U090xmp,I) @ rho_2 @ ((np.kron(U090xmp,I)).conj().T)                        # last pi/2
    res1 = (np.trace(irho_z@partial_trace(rho_3,2))).real                                       # NV state 0 population readout
    Sa.append(res1)

nn_r = 2*np.linspace(1,32,32)
#tau 구하기
index = Sa.index(min(Sa))
tau=t[index]
print(tau)

#여기까지는 mtop에 있는 코드와 같음
#즉, 위에 코드는 resornance frequency 를 찾기 위한 코드

#아래의 코드는 랜덤한 target state의 density rho를 2X2 -> 3X3로 변환하는 함수
def thtotw(ta):
    R = np.array([[ta[0,0],0,ta[0,1]],
                  [0,0,0],
                  [ta[1,0],0,ta[1,1]]])
    return R

#초기화
irho = np.kron(irho_z,irho_p)
rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T)


#원하는 STATE로 UNIVERSIAL GATE로 초기상태(UP)에서 STATE까지 가기위한 변수 구하기


def problem(vari):
    #최소 tau로 X-GATE의 N 먼저 찾기
    #x-gate
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[0]/2)@U_H)                                     #for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[0])@U_H)                                        # for tau
    rho_1=U_e2@irho@(U_e2.conj().T)                                                           # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                  # N & tau N개 생성
        rho_1 = U_e@np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # for N, tau
    rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # for 
    
#X-GATE에서의 N으로 Z-GATE의 TAU 찾기
#z-gate   
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)                                     # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)                                        # for tau
    rho_4=U_e2@rho_2@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                  # N & tau N개 생성
        rho_4 = U_e@np.kron(U180xp,I) @ rho_4 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # for N, tau
    rho_5 = U_e2 @ np.kron(U180xp,I) @ rho_4 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # for N & pi/2
    
    
    xob = (np.trace(Ix@partial_trace(rho_5,1))).real        # simulated result
    yob = (np.trace(Iy@partial_trace(rho_5,1))).real        # simulated result
    zob = (np.trace(Iz@partial_trace(rho_5,1))).real        # simulated result
    xxx = (np.trace(Ix@tr)).real                            # target state's
    yyy = (np.trace(Iy@tr)).real                            # target state's
    zzz = (np.trace(Iz@tr)).real                            # target state's
    cost = np.abs(xxx-xob)+np.abs(yyy-yob)+np.abs(zzz-zob)  # 0에 가까워야 함

    return cost


totres = [] #임의의 list 생성
#랜덤한 target state를 생성
for p in range (100): #100번 시행
    while 1 :
        U=rand_unitary(2)
        a=U[0,0]
        b=U[1,0]
        UU=np.array([[a],[b]])
        TU = UU.T
        CU = TU.conjugate()
        idden = UU*CU
        if np.abs(idden[1,1]) != 0:
            break
    tr=thtotw(idden) #3X3 density rho로 변환
    
    vari=[tau,1,0.1*tau]
    bounds = [(0.9*tau,1.1*tau),(1.0,33.0),(0.1*tau,0.8*tau)]
    # 아래의 세가지 method 중 사용할 method 빼고 주석 처리
    #res = optimize.minimize(problem,vari,bounds=bounds,method='Powell',options={'xtol':0.0001,'ftol':0.0001})  
    res = optimize.differential_evolution(problem,bounds=bounds,atol = 0.001)
    #res = optimize.dual_annealing(problem,bounds=bounds,maxiter=100)
    totres.append(res) #list에 결과들 입력

df = pd.DataFrame(totres)
df.to_csv('result.csv',index=False) #csv파일에 결과 저장
print('success')

