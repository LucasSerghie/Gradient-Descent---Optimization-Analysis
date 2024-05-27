import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data
X = np.array([(365, 28), (354, 28), (380, 53), (411, 61), (472, 88), (418, 60), (397, 55), (375, 35), (407, 61), (490, 100),
(403, 59), (423, 62), (398, 56), (475, 91), (338, 16), (475, 81), (343, 18), (386, 49), (360, 44), (363, 35),
(380, 44), (388, 58), (397, 62), (369, 50), (331, 16), (380, 49), (467, 78), (402, 53), (332, 24), (453, 87),
(400, 55), (444, 66), (365, 37), (412, 70), (457, 77), (406, 50), (319, 10), (425, 68), (322, 10), (390, 50),
(343, 26), (333, 29), (329, 19), (416, 64), (432, 77), (474, 99), (398, 52), (439, 77), (404, 62), (360, 26)])  # Features

y = np.array([
850, 785, 815, 914, 1288, 1040, 911, 778, 1021, 1230,
970, 1073, 1020, 1156, 774, 1157, 688, 872, 875, 923,
787, 922, 948, 798, 605, 886, 1214, 1004, 698, 1078,
991, 1070, 750, 1027, 1054, 968, 608, 953, 641, 821,
657, 772, 700, 996, 1052, 1231, 875, 1099, 1087, 709])  # Target variable

# Add bias term to features
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize weights for each method separately
W_const = np.zeros(len(X_b[0]))
W_lipschitz = np.zeros(len(X_b[0]))
W_adjusted = np.zeros(len(X_b[0]))

def predict(X, W):
    return X.dot(W)

def MSE(errors):
    return (1/(2 * len(errors))) * sum(error**2 for error in errors)

def gradientMSE(m, X, errors):
    return (1/m) * X.T.dot(errors)

def backtracking_line_search(X, W, y, grad, alpha, beta=0.5, sigma=0.1):
    while MSE(X.dot(W - alpha * grad) - y) > MSE(X.dot(W) - y) - sigma * alpha * np.linalg.norm(grad)**2:
        alpha *= beta
    return alpha

def compute_lipschitz_constant(m, X, y, a, b, num_points=10):
    x1_vals = np.linspace(a, b, num_points)
    x2_vals = np.linspace(a, b, num_points)
    y_vals = np.linspace(a, b, num_points)
    X1, X2, Y = np.meshgrid(x1_vals, x2_vals, y_vals)
    points = np.vstack([X1.ravel(), X2.ravel(), Y.ravel()]).T

    errors_matrix = np.apply_along_axis(lambda point: predict(X, point) - y, 1, points)
    gradients = np.apply_along_axis(lambda err: gradientMSE(m, X, err), 1, errors_matrix)

    grad_diff_matrix = np.linalg.norm(gradients[:, np.newaxis, :] - gradients[np.newaxis, :, :], axis=-1)
    point_diff_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)

    point_diff_matrix[point_diff_matrix == 0] = np.inf

    lipschitz_matrix = grad_diff_matrix / point_diff_matrix
    max_lipschitz = np.max(lipschitz_matrix)
    
    return max_lipschitz, points, gradients

def gradient_descent_const_step(X, y, W, learning_rate, iterations):
    m = len(y)
    optimal_values = []
    mse_values = []
    
    for iteration in range(iterations):
        predictions = predict(X, W)
        errors = predictions - y
        gradients = gradientMSE(m, X, errors)
        W -= learning_rate * gradients
        optimal_values.append(W.copy())
        mse_values.append(MSE(errors))
                    
    return W, optimal_values, mse_values

def gradient_descent_adjusted_step(X, y, W, learning_rate, iterations):
    m = len(y)
    optimal_values = []
    mse_values = []
    
    for iteration in range(iterations):
        predictions = predict(X, W)
        errors = predictions - y
        gradients = gradientMSE(m, X, errors)
        learning_rate = backtracking_line_search(X, W, y, gradients, learning_rate)
        W -= learning_rate * gradients
        optimal_values.append(W.copy())
        mse_values.append(MSE(errors))
                    
    return W, optimal_values, mse_values

# Parameters
a, b = -0.3, 0.3
iterations = 8  # Set the number of iterations to 8

# Compute Lipschitz constant learning rate
lipschitz_constant, lipschitz_points, lipschitz_gradients = compute_lipschitz_constant(len(y), X_b, y, a, b)
lipschitz_learning_rate = 1 / lipschitz_constant

# Compute gradient magnitudes and normalize
gradient_magnitudes = np.linalg.norm(lipschitz_gradients, axis=1)
normalized_magnitudes = (gradient_magnitudes - gradient_magnitudes.min()) / (gradient_magnitudes.max() - gradient_magnitudes.min())

# Set constant and initial adjusted learning rates
constant_learning_rate = 0.00001
adjusted_learning_rate = 5

# Train the models using different gradient descent methods
W_trained_with_constant_step, const_optimal_values, const_mse_values = gradient_descent_const_step(X_b, y, W_const, constant_learning_rate, iterations)
W_trained_with_lipschitz_step, lipschitz_optimal_values, lipschitz_mse_values = gradient_descent_const_step(X_b, y, W_lipschitz, lipschitz_learning_rate, iterations)
W_trained_with_adjusted_step, adjusted_optimal_values, adjusted_mse_values = gradient_descent_adjusted_step(X_b, y, W_adjusted, adjusted_learning_rate, iterations)

# Print the final weights for each method
print(f'W_trained_with_constant_step: {W_trained_with_constant_step}')
print(f'W_trained_with_lipschitz_step: {W_trained_with_lipschitz_step}')
print(f'W_trained_with_adjusted_step: {W_trained_with_adjusted_step}')

# Example of making a prediction
def make_prediction(new_data, W):
    new_data_b = np.c_[np.ones((new_data.shape[0], 1)), new_data]
    return predict(new_data_b, W)

# Example prediction
new_data = np.array([[6, 12]])
predicted_value_const = make_prediction(new_data, W_trained_with_constant_step)
predicted_value_lipschitz = make_prediction(new_data, W_trained_with_lipschitz_step)
predicted_value_adjusted = make_prediction(new_data, W_trained_with_adjusted_step)

print(f'Prediction with constant step: {predicted_value_const[0]}')
print(f'Prediction with Lipschitz step: {predicted_value_lipschitz[0]}')
print(f'Prediction with adjusted step: {predicted_value_adjusted[0]}')


# Create a figure with four subplots
fig = plt.figure(figsize=(28, 6))

# 3D Scatter plot
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Target')
ax1.set_title('3D Scatter Plot of Features and Target Variable')

# Plot the convergence of the optimal values
ax2 = fig.add_subplot(142)
ax2.plot(const_mse_values, label='Constant Step')
ax2.plot(lipschitz_mse_values, label='Lipschitz Step')
ax2.plot(adjusted_mse_values, label='Adjusted Step')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('MSE')
ax2.set_title('Convergence of Gradient Descent Methods')
ax2.legend()

# Plot the Lipschitz points with colors based on gradient magnitudes
ax3 = fig.add_subplot(143, projection='3d')
sc = ax3.scatter(lipschitz_points[:, 0], lipschitz_points[:, 1], lipschitz_points[:, 2], c=normalized_magnitudes, cmap='RdYlGn')
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('Y')
ax3.set_title('3D Scatter Plot of Lipschitz Points')
fig.colorbar(sc, ax=ax3, label='Gradient Magnitude')

# Plot the paths taken by gradient descent methods
ax4 = fig.add_subplot(144, projection='3d')

# Function to compute cost for given weights
def compute_cost(X, y, W):
    predictions = predict(X, W)
    errors = predictions - y
    return MSE(errors)

# Plot paths
colors = ['blue', 'orange', 'green']
labels = ['Constant Step', 'Lipschitz Step', 'Adjusted Step']
optimal_values = [const_optimal_values, lipschitz_optimal_values, adjusted_optimal_values]

for i, opt_vals in enumerate(optimal_values):
    weights = np.array(opt_vals)
    w1 = weights[:, 1]
    w2 = weights[:, 2]
    costs = [compute_cost(X_b, y, w) for w in weights]
    ax4.plot(w1, w2, costs, label=labels[i], color=colors[i])
    ax4.scatter(w1, w2, costs, color=colors[i])

ax4.set_xlabel('Weight 1 (w1)')
ax4.set_ylabel('Weight 2 (w2)')
ax4.set_zlabel('Cost')
ax4.set_title('Paths of Gradient Descent Methods')
ax4.legend()

plt.tight_layout()
plt.show()
