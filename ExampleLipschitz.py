import numpy as np
import matplotlib.pyplot as plt

X_new_1 = np.random.uniform(10, 100, 50).astype(int)

# Add noise to X_new[:, 1]
noise = np.random.uniform(-10, 15, 50).astype(int)
X_new_1_with_noise = X_new_1 + noise



# Choose slope and intercept for the linear relationship
a = 2  # Slope
b = 300  # Intercept

# Calculate X_new[:, 0] using the linear relationship
X_new_0 = a * X_new_1 + b
noise = np.random.uniform(-10, 5, 50).astype(int)
X_new_0_with_noise = X_new_0 + noise

print(len(X_new_1_with_noise), len(X_new_0))

# Stack X_new[:, 0] and X_new[:, 1] horizontally to form X_new
X_new = np.column_stack((X_new_0_with_noise, X_new_1_with_noise))

noise = np.random.randint(-110, 100, 50)
y_new = 2 * X_new[:, 0].astype(int) + 3 * X_new[:, 1].astype(int) + noise






# Print pairs data
for i in range(0, len(X_new), 10):
    print(', '.join([f"({X_new[j][0]}, {X_new[j][1]})" for j in range(i, i+10)]))
    
# Print singles data
for i in range(0, len(y_new), 10):
    print(', '.join([str(y_new[j]) for j in range(i, i+10)]))
    
    
    
# X = np.array([(102, 24), (180, 60), (96, 25), (134, 40), (130, 49), (100, 33), (192, 75), (148, 53), (88, 22), (168, 59),
# (230, 95), (76, 5), (188, 73), (128, 39), (172, 69), (158, 54), (144, 41), (146, 51), (100, 25), (204, 70),
# (92, 19), (194, 70), (166, 65), (92, 14), (184, 75), (122, 34), (126, 47), (70, 3), (110, 33), (140, 43),
# (132, 44), (96, 24), (166, 60), (190, 69), (122, 43), (92, 26), (192, 65), (182, 57), (78, 8), (212, 83),
# (130, 37), (142, 42), (114, 33), (176, 72), (164, 58), (236, 92), (228, 86), (116, 33), (196, 69), (164, 52)])  # Features

# y = np.array([
# 249, 495, 221, 427, 419, 293, 594, 498, 278, 522,
# 757, 196, 632, 390, 576, 469, 454, 474, 275, 588,
# 215, 573, 487, 254, 578, 338, 366, 114, 282, 457,
# 376, 247, 481, 616, 331, 276, 625, 511, 141, 698,
# 333, 456, 329, 530, 485, 760, 754, 309, 568, 495])  # Target variable

# print(len(X), len(y))


fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_new[:, 0], X_new[:, 1], y_new, c='blue', marker='o')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Target')
ax1.set_title('3D Scatter Plot of Features and Target Variable')

plt.show()