import numpy as np
import matplotlib.pyplot as plt

# Objective function: example quadratic function f(theta) = 0.5 * theta.T * A * theta - b.T * theta
def objective_function(theta):
    A = np.array([[3, 1], [1, 2]])
    b = np.array([1, 1])
    return 0.5 * theta.T @ A @ theta - b.T @ theta

# Gradient of the objective function
def gradient(theta):
    A = np.array([[3, 1], [1, 2]])
    b = np.array([1, 1])
    return A @ theta - b

# Gradient descent with a fixed step size using a known Lipschitz constant
def gradient_descent_fixed_lipschitz(initial_theta, L, max_iterations=1000, tolerance=1e-6):
    theta = initial_theta
    objective_values = []
    for t in range(max_iterations):
        grad = gradient(theta)
        if np.linalg.norm(grad) < tolerance:
            break
        alpha_t = 1 / L
        theta = theta - alpha_t * grad
        objective_values.append(objective_function(theta))
    return theta, objective_values

# Backtracking line search to adapt step size
def backtracking_line_search(theta, grad, learning_rate, beta=0.5, sigma=0.1):
    while objective_function(theta - learning_rate * grad) > objective_function(theta) - sigma * learning_rate * np.linalg.norm(grad)**2:
        learning_rate *= beta
    return learning_rate

# Gradient descent with adaptive step size using backtracking line search
def gradient_descent_adaptive_step(initial_theta, alpha_0, max_iterations=1000, tolerance=1e-6):
    theta = initial_theta
    alpha = alpha_0
    objective_values = []
    for t in range(max_iterations):
        grad = gradient(theta)
        if np.linalg.norm(grad) < tolerance:
            break
        print(f"Theta: {theta}, grad: {grad}, alpha: {alpha}")
        alpha = backtracking_line_search(theta, grad, alpha)
        theta = theta - alpha * grad
        objective_values.append(objective_function(theta))
    return theta, objective_values

# Gradient descent with a constant step size
def gradient_descent_constant_step(initial_theta, alpha, max_iterations=1000, tolerance=1e-6):
    theta = initial_theta
    objective_values = []
    for t in range(max_iterations):
        grad = gradient(theta)
        if np.linalg.norm(grad) < tolerance:
            break
        theta = theta - alpha * grad
        objective_values.append(objective_function(theta))
    return theta, objective_values

# Example usage
initial_theta = np.array([0.0, 0.0])
alpha_0 = 5.0  # initial step size

# Using fixed Lipschitz constant
L = 4  # example Lipschitz constant for the quadratic function
optimal_theta_fixed, objective_values_fixed = gradient_descent_fixed_lipschitz(initial_theta, L)

# Using backtracking line search
optimal_theta_backtracking, objective_values_backtracking = gradient_descent_adaptive_step(initial_theta, alpha_0)

# Using constant step size
alpha_constant = 0.1
optimal_theta_constant, objective_values_constant = gradient_descent_constant_step(initial_theta, alpha_constant)

# Plotting the convergence
plt.figure(figsize=(12, 6))
plt.plot(objective_values_fixed, label='Fixed Lipschitz Constant')
plt.plot(objective_values_backtracking, label='Backtracking Line Search')
plt.plot(objective_values_constant, label='Constant Step Size (0.1)')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence of Gradient Descent Methods')
plt.legend()
plt.grid(True)
plt.show()

print("Optimal theta with fixed Lipschitz constant:", optimal_theta_fixed)
print("Optimal theta with backtracking line search:", optimal_theta_backtracking)
print("Optimal theta with constant step size (0.1):", optimal_theta_constant)
