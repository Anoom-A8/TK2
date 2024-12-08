import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Log-loss function for y=1
def log_loss_y1(x):
    f = sigmoid(x)
    return -np.log(f)

# Log-loss function for y=0
def log_loss_y0(x):
    f = sigmoid(x)
    return -np.log(1 - f)