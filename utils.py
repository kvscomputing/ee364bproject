import numpy as np

'''
every group is the same, specified, size, assumed to be a divisor of n
'''

def generate_data(n, p, group_size, rho):
    cov_matrix = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

    X = np.random.multivariate_normal(np.zeros(p), cov_matrix, n)

    beta = np.random.normal(0, 1, p)

    Y = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)  # noise term
    sigma = np.linalg.norm(X @ beta) / np.sqrt(3)
    Y = X @ beta + sigma * epsilon
    
    groups = []
    for i in range(p//group_size):
        groups.append([j for j in range(i*group_size, (i+1)*group_size)])
    return X, Y, groups


def generate_group_matrices(groups):
    p = groups[-1][-1] + 1
    group_matrices = []
    for group in groups:
        group_matrix = np.zeros((len(group), p))
        for i in range(len(group)):
            group_matrix[i, group[i]] = 1
        group_matrices.append(group_matrix)
    return group_matrices

'''
precomputes XG^TGX^T for every group matrix G
'''

def precompute_log_barrier_terms(X, group_matrices):
    precomputations = []
    for i in range(len(group_matrices)):
        precomputations.append(X @ group_matrices[i].T @ group_matrices[i] @ X.T)
    return precomputations
    



