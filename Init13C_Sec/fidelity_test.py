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

def state_fidelity(rho_1, rho_2):
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        
        
print(state_fidelity(np.array([[0,0],[0,1]]), np.array([[1,0],[0,0]])))