import numpy as np
from qutip import *

# idden = rand_dm_ginibre(3, rank=1)
# print(idden)
mat1 = np.array([[9.99217967e-01-1.66533454e-16j, 0.00000000e+00+0.00000000e+00j,
                1.59679358e-04+7.97402070e-03j],
                [0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
                0.00000000e+00+0.00000000e+00j],
                [1.59679358e-04-7.97402070e-03j, 0.00000000e+00+0.00000000e+00j,
                7.82032532e-04-1.53672105e-16j]])

mat2 = mat1@mat1

mat3 = mat2.trace()
print(mat3)

# def matrix_mult(A, B):
#     temp = [[0] * (len(A)) for _ in range(len(B[0]))]
#     for i in range(len(A)):
#         for j in range(len(A[0])):
#             for k in range(len(B[0])):
#                 temp[i][k] += A[i][j] * B[j][k]
#     return temp


# def matrix_pow(A, n):
#     if n == 1:
#         return A
#     if n % 2 == 0:
#         temp = matrix_pow(A, n//2)
#         return matrix_mult(temp, temp)
#     else:
#         temp = matrix_pow(A, n-1)
#         return matrix_mult(temp, A)


# mat4 = matrix_pow(mat1, 2)
# mat5 = mat4.trace()

# print(mat5)