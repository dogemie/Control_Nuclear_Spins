# import numpy as np

# def angles_to_density_matrix(theta, phi):
#     state_vector = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
#     density_matrix = np.outer(state_vector, np.conj(state_vector))
#     return density_matrix

# # Example usage
# theta = 2.20893233455532
# phi = 0.785398163397448
# rho = angles_to_density_matrix(theta, phi)
# print(rho)

import numpy as np

def rotation_matrix(theta, phi):
    # Define the rotation matrix R(θ, φ)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R = np.array([[cos_theta, 0, sin_theta],
                  [0, 1, 0],
                  [-sin_theta, 0, cos_theta]])
    R = np.dot(R, np.array([[cos_phi, sin_phi, 0],
                            [-sin_phi, cos_phi, 0],
                            [0, 0, 1]]))
    return R

# Define the matrix [[1, 0], [0, 0]]
M = np.array([[1, 0], [0, 0]])

theta = np.pi/4
phi = np.pi/3

# Define the transpose of the matrix P(θ)
Pt = np.array([[1, 0, 0],
              [0, np.cos(theta), -np.sin(theta)],
              [0, np.sin(theta), np.cos(theta)]])

# Define the rotation angles theta and phi


# Compute the rotation matrix R(θ, φ)
R = rotation_matrix(theta, phi)

# Compute the result R(φ) R(θ) [[1, 0], [0, 0]] P(θ)ᵀ R(φ)ᵀ
result = np.dot(R, np.dot(M, np.dot(Pt.T, np.dot(R.T, M))))

# Print the result
print(result)