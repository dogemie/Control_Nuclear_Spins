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

for x in range(10):
    deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)] #초기값을 넣는 랜덤변수
    print(deg)