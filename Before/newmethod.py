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

tau = 0.52
#여기까지는 mtop에 있는 코드와 같음
#즉, 위에 코드는 resornance frequency 를 찾기 위한 코드

#아래의 코드는 랜덤한 target state의 density rho를 2X2 -> 3X3로 변환하는 함수
def thtotw(ta):
    R = np.array([[ta[0,0],0,ta[0,1]],
                  [0,0,0],
                  [ta[1,0],0,ta[1,1]]])
    return R

#초기화
irho = np.kron(irho_z,irho_MIX)
rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T)
ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz)
eigvals = np.linalg.eigh(ham)[0] # diagonalizing the Hamiltonian 
eigvecs = -1*np.linalg.eigh(ham)[1]
E = np.diag(eigvals)             # exponent of eigenvalues
U_H= eigvecs.conj().T  
t = np.linspace(0.2,0.225,1000)
#원하는 STATE로 UNIVERSIAL GATE로 초기상태(UP)에서 STATE까지 가기위한 변수 구하기
c=len(t)

Sa = []
Sb = []
Sc = []
Se = []
    #최소 tau로 X-GATE의 N 먼저 찾기
    #x-gate
#for e Ry(pi/2)
for a in range(c):    
    rho1 = np.kron(U090yp,I)@irho@(np.kron(U090yp,I).conj().T)                              # Ry 90도

    #for N Rx(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E* 0.52853868/2)@U_H)                                  # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E* 0.52853868)@U_H)                                     # for tau
    rho2=U_e2@rho1@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*9):                                                # N과 tau를 N개 생성
        rho2 = U_e@np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho3 = U_e2 @ np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for e Rx(pi/2)
    rho4 = np.kron(U090xp,I)@rho3@(np.kron(U090xp,I).conj().T)                              # Rx 90도

    #for N Rz(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E*t[a])@U_H)                                   # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E*t[a])@U_H)                                      # for tau/2
    rho5=U_e2@rho4@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*18):                                                # N과 tau를 N개 생성
        rho5 = U_e@np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho6 = U_e2 @ np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for N Rx(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E* 0.52853868/2)@U_H)                                  # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E* 0.52853868)@U_H)                                     # for tau
    rho7=U_e2@rho6@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*9):                                                # N과 tau를 N개 생성
        rho7 = U_e@np.kron(U180xp,I) @ rho7 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho8 = U_e2 @ np.kron(U180xp,I) @ rho7 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    xx = (np.trace(Ix@partial_trace(rho8,1))).real
    yy = (np.trace(Iy@partial_trace(rho8,1))).real
    zz = (np.trace(Iz@partial_trace(rho8,1))).real

    Sa.append(xx)
    Sb.append(yy)
    Sc.append(np.abs(xx)+np.abs(yy))
    Se.append(zz)
Sd = []
for p in range(1000):
    Sd.append(0)
print(min(Sc))

print(Sc.index(min(Sc)))
plt.plot(t,Sa)
plt.plot(t,Sb)
#plt.plot(t,Se)
plt.plot(t,Sd)
# %%
