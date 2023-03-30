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

csv = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Result_20230328_174201.csv')

csv_fin = pd.DataFrame()
csv_fin2 = pd.DataFrame()
csv_fin = pd.concat([csv], axis=0)



print(csv_fin['idden'][1:-1])