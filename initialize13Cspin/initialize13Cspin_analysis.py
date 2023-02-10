#%%
# initialization system
# 이 코드는 13C의 nuclear spin을 초기화 시키는 코드입니다.
# 각 gate의 파라미터를 찾아서 초기화 시킵니다.
# initial:mixed state, target : up state
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
import time
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  

def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = np.array([[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]])
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
Szp  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
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
U180yp = UO(1,0,pi/2,pi,0)
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

#rotation operator(not use at this code)
def Rx(theta):
    Rx = e**(1j*theta/2*sx)
    return Rx

def Ry(theta):
    Ry = e**(1j*theta/2*sy)
    return Ry

def Rz(phi):
    Rz = e**(1j*phi/2*sz)
    return Rz 

irho = np.kron(irho_p,irho_MIX)



tau = 0.52 #본 코드에서는 tau를 찾는 과정을 생략하였습니다. 그러므로 tau를 지정하여 줍니다.
# tau = random.random()

ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz)
eigvals = np.linalg.eigh(ham)[0] # diagonalizing the Hamiltonian 
eigvecs = -1*np.linalg.eigh(ham)[1]
E = np.diag(eigvals)             # exponent of eigenvalues
U_H= eigvecs.conj().T  


def problem(vari):
    #for e Ry(pi/2)
    rho1 = np.kron(U090yp,I)@irho@(np.kron(U090yp,I).conj().T)                              # Ry 90도

    #for N Rx(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E* vari[0]/2)@U_H)                                  # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E* vari[0])@U_H)                                     # for tau
    rho2=U_e2@rho1@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                # N과 tau를 N개 생성
        rho2 = U_e@np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho3 = U_e2 @ np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for e Rx(pi/2)
    rho4 = np.kron(U090xp,I)@rho3@(np.kron(U090xp,I).conj().T)                              # Rx 90도

    #for N Rz(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)                                   # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)                                      # for tau/2
    rho5=U_e2@rho4@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                # N과 tau를 N개 생성
        rho5 = U_e@np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho6 = U_e2 @ np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for N Rx(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E* vari[0]/2)@U_H)                                  # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E* vari[0])@U_H)                                     # for tau
    rho7=U_e2@rho6@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                # N과 tau를 N개 생성
        rho7 = U_e@np.kron(U180xp,I) @ rho7 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho8 = U_e2 @ np.kron(U180xp,I) @ rho7 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    # projection&trace
    xob = (np.trace(Sxp@partial_trace(rho8,2))).real # for e spin
    yob = (np.trace(Syp@partial_trace(rho8,2))).real 
    zob = (np.trace(Szp@partial_trace(rho8,2))).real

    xx = (np.trace(Ix@partial_trace(rho8,1))).real # for N spin
    yy = (np.trace(Iy@partial_trace(rho8,1))).real
    zz = (np.trace(Iz@partial_trace(rho8,1))).real
    
    # print(xx,yy,zz)

    cost = np.abs(0-xx)+np.abs(0-yy)+np.abs(1-zz)
    return cost

#결과들을 저장할 list 생성
aa= []
bb= []
cc= []
dd= []

tol = 1e-8

for p in range(3):
    vari=[tau,9,0.1*tau]  #초기값
    bounds = [(0.85*tau,1.15*tau),(1.0,17.0),(0.05*tau,0.8*tau)] #boundary
    start = time.time()
    res4 = optimize.shgo(problem,bounds=bounds,iters=4,options={'xtol':tol,'ftol':tol}) #SHGO method
    dd.append(res4)
    end = time.time()
    final = end - start
    print(final)

date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)

# 결과들을 list에 저장하여 csv파일로 저장
df4 = pd.DataFrame(dd)

df4.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/shgo' + printdate + '.csv',index=False)
print('success')

# # %%
# print(U090xp)
# print(U090yp)
# # %%
