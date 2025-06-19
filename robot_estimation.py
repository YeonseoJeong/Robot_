import numpy as np
import matplotlib.pyplot as plt

# 1. Problem parameters
theta_true = np.array([2.0, 3.0])  # True position
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])  # Noise covariance

def generate_measurements(theta, Sigma, n):
    """Generate n noisy 2D measurements."""
    return np.random.multivariate_normal(theta, Sigma, size=n)

def mvu_estimator(zs):
    """MVU estimator for mean with known covariance and i.i.d. noise."""
    return np.mean(zs, axis=0)

# 2. Experiment setup
n_trials = 1000
n_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
empirical_variances = []

for n in n_list:
    estimates = []
    for _ in range(n_trials):
        zs = generate_measurements(theta_true, Sigma, n)
        theta_hat = mvu_estimator(zs)
        estimates.append(theta_hat)
    estimates = np.array(estimates)
    var = np.cov(estimates.T)
    empirical_variances.append(var)

# 3. Plotting

# (1) One trial: measurements and true pose
zs = generate_measurements(theta_true, Sigma, 50)
plt.figure(figsize=(6, 6))
plt.scatter(zs[:, 0], zs[:, 1], label='Measurements', alpha=0.6)
plt.scatter([theta_true[0]], [theta_true[1]], color='red', label='True Pose', marker='x', s=100)
plt.title('Noisy Measurements (one trial)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# (2) Distribution of MVU estimates
estimates = []
for _ in range(n_trials):
    zs = generate_measurements(theta_true, Sigma, 20)
    theta_hat = mvu_estimator(zs)
    estimates.append(theta_hat)
estimates = np.array(estimates)
plt.figure(figsize=(6, 6))
plt.scatter(estimates[:, 0], estimates[:, 1], alpha=0.5, label='MVU Estimates')
plt.scatter([theta_true[0]], [theta_true[1]], color='red', label='True Pose', marker='x', s=100)
plt.title('Distribution of MVU Estimates (n=20)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# (3) Plot empirical variance of MVU estimator
emp_var_x = [var[0, 0] for var in empirical_variances]
emp_var_y = [var[1, 1] for var in empirical_variances]

plt.figure()
plt.plot(n_list, emp_var_x, 'o-', label='Var(x)')
plt.plot(n_list, emp_var_y, 's-', label='Var(y)')
plt.title('Variance of MVU Estimator vs. Number of Samples')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Empirical Variance')
plt.legend()
plt.grid(True)
plt.show()
