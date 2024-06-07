import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from utils import generate_data, generate_group_matrices, precompute_log_barrier_terms
from methods import admm, newton, truncated_newton, truncated_newton_limited_hessian, group_lasso_penalty, bfgs_newton, admm_low_dimensional
import time

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
NUM_ITERS = 10000
FSTAR = 49.9649387126726

X_TRAIN_PATH = "./data/X_train.csv"
Y_TRAIN_PATH = "./data/Y_train.csv"

def load_data(X_train_path, y_train_path):
    X_train = np.loadtxt(X_train_path, delimiter=",")
    X_train = np.hstack([np.ones(X_train.shape[0])[:, np.newaxis], X_train])
    y_train = np.loadtxt(y_train_path, delimiter=",")
    return X_train, y_train

X, y = load_data(X_TRAIN_PATH, Y_TRAIN_PATH)

alpha = np.zeros(X.shape[1])
beta = np.zeros(X.shape[1])
gamma = np.zeros(X.shape[1])

beta, alpha, gamma, obj_vals = admm_low_dimensional(X, y, GROUPS, RHO, NUM_ITERS, alpha, beta, gamma, 8)
plt.plot(obj_vals, label="ADMM")
plt.legend()
plt.savefig("admm_low_dimensional.png")
