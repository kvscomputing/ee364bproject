import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

LAMBDA = 0.02
RHO = 0.4
T = 10
GROUPS = [
    [0],
    [1],
    [2],
    [3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15],
    [16],
    [17],
    [18],
]
NUM_ITERS = 10000
FSTAR = 49.9649387126726

X_TRAIN_PATH = "./data/X_train.csv"
Y_TRAIN_PATH = "./data/Y_train.csv"

def load_data(X_train_path, y_train_path):
    X_train = np.loadtxt(X_train_path, delimiter=",")
    X_train = np.hstack([np.ones(X_train.shape[0])[:, np.newaxis], X_train])
    y_train = np.loadtxt(y_train_path, delimiter=",")
    return X_train, y_train

X_train, y_train = load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
print(X_train.shape)
print(y_train.shape)
