import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# True robot position (unknown in practice)
theta_true = np.array([4.5, 6.0])

# Landmark positions (m landmarks)
landmarks = np.array([
    [2.0, 3.0],
    [8.0, 2.0],
    [5.0, 8.0],
    [1.0, 7.0]
])

m = landmarks.shape[0]
sigma = 1.0  # Standard deviation of noise

# Simulate noisy distance measurements
def simulate_measurements(theta, landmarks, sigma):
    true_distances = np.linalg.norm(landmarks - theta, axis=1)
    noise = np.random.normal(0, sigma, size=true_distances.shape)
    return true_distances + noise

z = simulate_measurements(theta_true, landmarks, sigma)

# Negative log-likelihood function for MLE
def neg_log_likelihood(theta, landmarks, z, sigma):
    theta = np.array(theta)
    predicted_distances = np.linalg.norm(landmarks - theta, axis=1)
    residuals = z - predicted_distances
    nll = 0.5 * np.sum((residuals / sigma) ** 2)
    return nll

# Initial guess for robot position
theta_init = np.mean(landmarks, axis=0)

# Perform MLE using scipy.optimize.minimize
result = minimize(
    neg_log_likelihood,
    theta_init,
    args=(landmarks, z, sigma),
    method='L-BFGS-B'
)
theta_mle = result.x

# Plotting
plt.figure(figsize=(7, 7))
plt.title("Robot 2D Localization via MLE")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Plot landmarks
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b', s=100, label='Landmarks')

# Plot true robot position
plt.scatter(theta_true[0], theta_true[1], c='g', marker='x', s=100, label='True position')

# Plot MLE estimate
plt.scatter(theta_mle[0], theta_mle[1], c='r', s=100, label='MLE estimate')

# Plot measured circles
for i in range(m):
    circle = plt.Circle(landmarks[i], z[i], color='gray', fill=False, linestyle='dashed', alpha=0.5)
    plt.gca().add_patch(circle)

plt.legend()
plt.axis('equal')
plt.xlim(-2.5, 12.5)
plt.ylim(-5, 12.5)
plt.show()