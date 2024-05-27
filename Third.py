import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# X = np.array([
#       (123, 0.78), (456, 0.12), (789, 0.95), (231, 0.34), (567, 0.82), (100, 0.41), (345, 0.09), (678, 0.65), (212, 0.27), (543, 0.58), (876, 0.16), (101, 0.73), (432, 0.49), (765, 0.03), (987, 0.89), (234, 0.62), (560, 0.24), (891, 0.51), (120, 0.10), (450, 0.37), (780, 0.92), (109, 0.70), (438, 0.46), (762, 0.06),
#       (984, 0.86), (215, 0.56), (546, 0.18), (873, 0.43), (118, 0.79), (447, 0.21), (774, 0.80), (105, 0.67), (435, 0.31), (768, 0.01), (990, 0.98), (221, 0.53), (552, 0.14), (885, 0.40), (114, 0.76), (443, 0.19), (771, 0.77), (102, 0.64), (431, 0.29), (760, 0.04), (981, 0.91), (227, 0.50), (558, 0.11), (889, 0.38),
#       (111, 0.74), (439, 0.17), (998, 0.83), (229, 0.47), (555, 0.13), (882, 0.44), (117, 0.71), (444, 0.23), (777, 0.78), (108, 0.61), (437, 0.26), (764, 0.07), (986, 0.87), (219, 0.54), (342, 0.97), (675, 0.35), (901, 0.08)
#     ])

# y = np.array([
#  256, 1012, 1568, 562, 1134, 240, 870, 1406, 432, 1064, 
#  1752, 242, 968, 1496, 1924, 568, 1100, 1632, 340, 970, 
#  1500, 270, 936, 1468, 1900, 492, 1024, 1556, 316, 952, 
#  1484, 220, 908, 1440, 1872, 524, 1056, 1588, 372, 996, 
#  1528, 248, 924, 1456, 1888, 1996, 784, 1316, 1848, 676,
#  1208, 1740, 508, 1040, 1572, 2028, 816, 1348, 1880, 712,
#  1244, 1776, 540, 1072, 1604])

X = np.array([(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)])  # Feature 1
y = np.array([5, 9, 12, 18, 20])  # Target variable


X_b = np.c_[np.ones((X.shape[0], 1)), X]

W = np.zeros(3)


def predict(X, W):
    return X.dot(W)

def MSE(errors):
    return (1/(2 * len(errors))) * sum(error**2 for error in errors)

# Define the function
def gradientMSE(m, X, errors):
    return 1/m * X.T.dot(errors)

def backtracking_line_search(X, W, y, grad, alpha, beta=0.5, sigma=0.1):
    while MSE(X.dot(W - alpha * grad) - y) > MSE(X.dot(W) - y) - sigma * alpha * np.linalg.norm(grad)**2:
        alpha *= beta
    return alpha

def compute_lipschitz_constant(m, X, y,  a, b, num_points=10):
    x1_vals = np.linspace(a, b, num_points)
    x2_vals = np.linspace(a, b, num_points)
    y_vals = np.linspace(a, b, num_points)
    X1, X2, Y = np.meshgrid(x1_vals, x2_vals, y_vals)
    points = np.vstack([X1.ravel(), X2.ravel(), Y.ravel()]).T
    

        # Compute predictions and errors for all points in the grid
    errors_matrix = np.apply_along_axis(lambda point: predict(X, point) - y, 1, points)

    # Compute gradients for all points in the grid
    gradients = np.apply_along_axis(lambda err: gradientMSE(m, X, err), 1, errors_matrix)

    # Create matrices for pairwise differences
    grad_diff_matrix = np.linalg.norm(gradients[:, np.newaxis, :] - gradients[np.newaxis, :, :], axis=-1)
    point_diff_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)

    # Avoid division by zero by setting zero distances to infinity
    point_diff_matrix[point_diff_matrix == 0] = np.inf

    # Compute the Lipschitz ratios
    lipschitz_matrix = grad_diff_matrix / point_diff_matrix

    # Find the maximum Lipschitz ratio
    max_lipschitz = np.max(lipschitz_matrix)
    
    return max_lipschitz


def gradient_descent_const_step(X, y, W, learning_rate, iterations):
    m = len(y)
    optimal_values = []
    
    for iteration in range(iterations):
        # Calculate predictions
        predictions = predict(X, W)
        
        # Calculate the error
        errors = predictions - y

        gradients = gradientMSE(m, X, errors)

        # Update the weights
        W -= learning_rate * gradients
        optimal_values.append(W)

                    
    return W, optimal_values


def gradient_descent_adjusted_step(X, y, W, learning_rate, iterations):
    m = len(y)
    optimal_values = []
    
    for iteration in range(iterations):
        # Calculate predictions
        predictions = predict(X, W)
        
        # Calculate the error
        errors = predictions - y

        gradients = gradientMSE(m, X, errors)
    

        learning_rate = backtracking_line_search(X, W, y, gradients, learning_rate)

        # Update the weights
        W -= learning_rate * gradients
        optimal_values.append(W)
        
                    
    return W, optimal_values


a, b = -0.3, 0.3
iterations = 100

lipschitz_learning_rate = compute_lipschitz_constant(len(y), X_b, y, a, b)

constant_learning_rate = 0.01

adjusted_learning_rate = 5


# Train the model
W_trained_with_constant_step, const_optimal_values = gradient_descent_const_step(X_b, y, W, constant_learning_rate, iterations)
# W_trained_with_lipschitz_step, lipschitz_optimal_values = gradient_descent_const_step(X_b, y, W, lipschitz_learning_rate, iterations)
# W_trained_with_adjusted_step, adjusted_optimal_values = gradient_descent_adjusted_step(X_b, y, W, adjusted_learning_rate, iterations)


# Print the final weights
print(f'W_trained_with_constant_step: {W_trained_with_constant_step}')
# print(f'W_trained_with_lipschitz_step: {W_trained_with_lipschitz_step}')
# print(f'W_trained_with_adjusted_step: {W_trained_with_adjusted_step}')


# def make_prediction(new_data, W):
#     new_data_b = np.c_[np.ones((new_data.shape[0], 1)), new_data]
#     return predict(new_data_b, W)

# # Example of making a prediction
# new_data = np.array([[6, 12]])
# predicted_value = make_prediction(new_data, W_trained)

# print(f'Prediction for {new_data[0][0]} {new_data[0][1]}: {predicted_value[0]}')
