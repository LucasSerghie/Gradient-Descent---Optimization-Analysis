import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define your dataset manually
x1 = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245])  # Feature 1
x2 = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9])  # Feature 2
y = np.array([0.5,2.0,4.5,8.0,12.5,18.0,24.5,32.0,40.5,50.0,60.5,72.0,84.5,98.0,112.5,128.0,144.5,162.0,180.5,200.0,220.5,242.0,264.5,288.0,312.5,338.0,364.5,392.0,420.5,450.0,480.5,512.0,544.5,578.0,612.5,648.0,684.5,722.0,760.5,800.0,840.5,882.0,924.5,968.0,1012.5,1058.0,1104.5,1152.0,1200.5])  # Target variable

# Step 2: Feature Scaling (Normalization)
def normalize(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized

X_normalized = normalize(np.column_stack((x1, x2)))

# Step 3: Define the mean squared error (MSE) function as the objective function
def mean_squared_error(theta, X, y):
    m = len(y)
    y_pred = X.dot(theta)
    mse = np.sum((y_pred - y) ** 2) / (2 * m)
    return mse

# Step 4: Modify the gradient function for the MSE function
def gradient_mse(theta, X, y):
    m = len(y)
    grad = X.T.dot(X.dot(theta) - y) / m
    return grad


# Step 5: Implement gradient descent using the MSE function
def gradient_descent_mse(X, y, initial_theta, learning_rate, max_iterations=1000, tolerance=1e-6):
    theta = initial_theta
    objective_values = []
    for _ in range(max_iterations):
        grad = gradient_mse(theta, X, y)
        if np.linalg.norm(grad) < tolerance:
            break
        theta = theta - learning_rate * grad
        objective_values.append(mean_squared_error(theta, X, y))
    return theta, objective_values

# Initial parameters
initial_theta = np.random.randn(3, 1)
learning_rate = 0.1

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# Run gradient descent using MSE
optimal_theta, objective_values = gradient_descent_mse(X_b, y, initial_theta, learning_rate)

# New values for x1 and x2
new_x1 = 70
new_x2 = 1.4

# Compute prediction using optimal theta values
prediction = optimal_theta[0] + optimal_theta[1] * new_x1 + optimal_theta[2] * new_x2

# Print prediction
print("Prediction:", prediction)


# Plot the convergence
plt.figure(figsize=(10, 6))
plt.plot(objective_values)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Convergence of Gradient Descent with MSE (Normalized Features)')
plt.grid(True)
plt.show()
