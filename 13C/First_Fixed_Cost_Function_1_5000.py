#%%
# initialization system
# 이 코드는 13C의 nuclear spin을 초기화 시키는 코드입니다.,·
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
import random
from mpl_toolkits.mplot3d import Axes3D
from sys import stdout
from tqdm import tqdm
from tqdm import trange
from scipy.linalg import fractional_matrix_power

totalstart = time.time() #시작 시간 저장

#Generating gate function
def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]
    return UA

## Define dimension, pauli matrices
i   = 1j #1j
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/i*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
#sz  = [1, 0, 0; 0, -1, 0; 0, 0, 0]
I   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Rotation matrix projected into 2 level system
Sxp  = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sxm  = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Syp  = 1/i*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
Sym  = 1/i*np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
Szp  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -i],[0, 0, 0], [i, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])

# Pauli basis for 13C nuclear spin
Ix  = 1/2*np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/2/i*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
 

## Define sweep parameters
Sweep = 1001
N = Sweep
B = 403 #[G] magnetic field

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

# Define initial state of the system

irho_p = np.array([[1,0,0],[0,0,0],[0,0,0]]) #[0,0,0;0,0,0]

irho_m = np.array([[0,0,0],[0,0,0],[0,0,1]]) #[0,0,0;0,0,1]

irho_z = np.array([[0,0,0],[0,1,0],[0,0,0]]) #[0,1,0;0,0,0]

irho_mix = np.array([[1/2,0,0],[0,1/2,0],[0,0,0]]) #[1/2,0,0;0,1/2,0;0,0,0]

irho_Z = np.array([[0,0,0],[0,0,0],[0,0,1]]) #target state

irho_MIX = np.array([[1/2,0,0],[0,0,0],[0,0,1/2]])

irho = np.kron(irho_z,irho_MIX) #initial state
trace = [1, 1, 0, 100] # trace of the X, Y, Z, and total density matrices




def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)

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

        #for N Rz(pi/2) //이부분이 Z pulse를 다루고 있다면 N을 따로 분리해야하나?>
        U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)                                   # for tau/2
        U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)                                      # for tau/2
        rho5=U_e2@rho4@(U_e2.conj().T)                                                          # first tau/2
        for k in range(1,2*math.trunc(vari[3])):                                                # N과 tau를 N개 생성
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
        
        cost = 1 - state_fidelity(irho_Z, partial_trace(rho8, 1))
        
        
        # if(np.abs(xx) < np.abs(trace[0])):
        #     trace[0] = xx
        # if(np.abs(yy) < np.abs(trace[1])):
        #     trace[1] = yy
        # if(zz < trace[2]):
        #     trace[2] = zz
            
        cost2 = 2 * vari[0] * vari[1] + vari[2] * vari[3]
        
        if(cost2 < trace[3]):
            trace[3] = cost2
            trace[0] = xx
            trace[1] = yy
            trace[2] = zz
        
        weight = 1/5000
        fin_cost = cost + weight * cost2
        # print("cost: " + str(cost) + " cost2: " + str(cost2) + " fin_cost: " + str(fin_cost) + " x: " + str(xx) + " y: " + str(yy) + " z: " + str(zz) + " min z: " + str(trace[2]))
        return fin_cost
        
aa = []
dd = []
count = 1
tot_sum = 0


for ccc in tqdm(range(10)): # range 번의 실험을 진행한다.
    trace = [1, 1, 0, 100]
    start = time.time()
    #for making 13C nuclear random dataset
    gammaN = 2*pi*1.071e-3 #[MHz/G]
    Al    = 2*pi * random.uniform(0.05, 0.8) #[MHz] # A_|| hyperfine term
    Ap = 2*pi* random.uniform(0.05, 0.3) #[MHz] # A_per hyperfine term

    #Initialization
    rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) # superposition state on NV

    Sa= []

    for h in range(N): # N개의 pulse를 생성한다.
        ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz) # Hamiltonian
        
        eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
        eigvecs = -1*np.linalg.eigh(ham)[1]         # eigenvectors
        E = np.diag(eigvals)                        # exponent of eigenvalues
        U_H= eigvecs.conj().T                       # unitary matrix formed by eigenvectors

        
        #free evolution unitary operator
        U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*t[h]/2)@U_H) # for tau/2
        U_e  = (U_H.conj().T)@(linalg.expm(-i*E*t[h])@U_H)  # for tau
        rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)                  # first tau/2
        for k in range(n-1):                                   # N과 tau를 N개 생성
            rho_1 = U_e @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
            
        rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2
        rho_3 = np.kron(U090xmp,I) @ rho_2 @ ((np.kron(U090xmp,I)).conj().T)    # last pi/2
        res1 = (np.trace(irho_z@partial_trace(rho_3,2))).real                   # NV state 0 population readout
        Sa.append(res1)                                                       # append to list
        
    index = Sa.index(min(Sa)) # list에서 가장 작은 값을 가지는 index를 찾는다.
    tau=t[index] # 그 index에 해당하는 tau를 찾는다.

    ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz) # Hamiltonian
    eigvals = np.linalg.eigh(ham)[0] # diagonalizing the Hamiltonian 
    eigvecs = -1*np.linalg.eigh(ham)[1] # eigenvectors
    E = np.diag(eigvals)             # exponent of eigenvalues
    U_H= eigvecs.conj().T         # unitary matrix formed by eigenvectors

    xx=0
    yy=0
    zz=0

    #결과들을 저장할 list 생성
    

    tol = 1e-8 #tolerance

    for p in range(1): # 1번의 실험을 진행한다.(지역 최적화 알고리즘을 사용할 경우에 수정한다.)
        vari=[tau,9,0.1*tau, 9]  #초기값
        bounds = [(0.95*tau,1.05*tau),(1.0,5.0),(0.00000000000000000001*tau,0.5*tau),(1.0,25.0)] #boundary
        
        res4 = optimize.shgo(problem,bounds=bounds,iters=5,options={'maxfev': 750, 'maxiter': 750, 'xtol':1e-15,'ftol':1e-17}) #SHGO method
        res4['x'][1] = round(res4['x'][1]) #rounding
        res4['x'][3] = round(res4['x'][3]) #rounding

        if(np.abs(res4['fun']) < 0.1): #fidelity가 0.05보다 작으면 성공
            dd.append([Al, Ap, res4['x'][0], res4['x'][1], res4['x'][2], res4['x'][3], res4['fun'], trace[0], trace[1], trace[2], res4['nfev'], tau, "shgo"])
            end = time.time()
            final = end - start
            count = count + 1
        else: #fidelity가 0.05보다 크면 실패
            res4 = optimize.shgo(problem,bounds=bounds,iters=5,options={'maxfev': 2500, 'maxiter': 2500, 'xtol':1e-15,'ftol':1e-17}) #SHGO method
            res4['x'][1] = round(res4['x'][1]) #rounding
            res4['x'][3] = round(res4['x'][3]) #rounding
            if(np.abs(res4['fun']) < 0.1): #fidelity가 0.05보다 작으면 성공
                dd.append([Al, Ap, res4['x'][0], res4['x'][1], res4['x'][2], res4['x'][3], res4['fun'], trace[0], trace[1], trace[2], res4['nfev'], tau, "shgo"])
                end = time.time()
                final = end - start
                count = count + 1
            else:
                res4 = optimize.shgo(problem,bounds=bounds,iters=6,options={'maxfev': 5000, 'maxiter': 5000, 'xtol':1e-15,'ftol':1e-17}) #SHGO method
                res4['x'][1] = round(res4['x'][1]) #rounding
                res4['x'][3] = round(res4['x'][3]) #rounding
                if(np.abs(res4['fun']) < 0.1): #fidelity가 0.05보다 작으면 성공
                    dd.append([Al, Ap, res4['x'][0], res4['x'][1], res4['x'][2], res4['x'][3], res4['fun'], trace[0], trace[1], trace[2], res4['nfev'], tau, "shgo"])
                    end = time.time()
                    final = end - start
                    count = count + 1
                else:
                    aa.append([Al, Ap, res4['x'][0], res4['x'][1], res4['x'][2], res4['x'][3], res4['fun'], trace[0], trace[1], trace[2], res4['nfev'], tau, "shgo"])
                    end = time.time()
                    final = end - start
                    count = count + 1
stdout.write("\n")

    
date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)

# 결과들을 list에 저장하여 csv파일로 저장
df4 = pd.DataFrame(dd)
df4.rename(columns={0:"Al", 1:"Ap", 2: "Xtau", 3: "XN", 4: "Ztau", 5: "ZN", 6: "fun", 7: "xx", 8: "yy", 9: "zz", 10: "fev"}, inplace=True)
df4.to_csv('C:/Users/KIST3/Desktop/13C/first_test/AA5000pass' + printdate + '.csv',index=False)

df2 = pd.DataFrame(aa)
df2.rename(columns={0:"Al", 1:"Ap", 2: "Xtau", 3: "XN", 4: "Ztau", 5: "ZN", 6: "fun", 7: "xx", 8: "yy", 9: "zz", 10: "fev"}, inplace=True)
df2.to_csv('C:/Users/KIST3/Desktop/13C/first_test/BB5000fail' + printdate + '.csv',index=False)
print('success')

totalend = time.time()
print(totalend - totalstart)

# # %%