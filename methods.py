import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from functools import partial
import cvxpy as cp
import time

'''
Groups is given as in HW5.
'''
def group_lasso_prox(v, lambd, groups):
    proximal_operator = np.zeros(v.shape)
    for group in groups:
        proximal_operator[group] = np.maximum(0, 1 - lambd/np.linalg.norm(v[group], ord=2)) * v[group]
    return proximal_operator

def affine_projection(X, y, v):
    return v - X.T @ np.linalg.inv(X @ X.T) @ (X @ v - y)

def group_lasso_penalty(beta, groups):
    penalty = 0
    for group in groups:
        penalty += np.linalg.norm(beta[group], ord=2)
    return penalty

def admm(X, y, groups, rho, num_steps, alpha_init, beta_init, gamma_init):
    start_time = time.time()
    times = []
    alpha = alpha_init
    beta = beta_init
    gamma = gamma_init
    obj_vals = np.zeros(num_steps)
    for k in range(num_steps):
        beta = group_lasso_prox(alpha - gamma, 1/rho, groups)
        alpha = affine_projection(X, y, beta + gamma)
        gamma = gamma + beta - alpha
        obj_vals[k] = group_lasso_penalty(beta, groups)
        times.append(time.time() - start_time)
    return beta, alpha, gamma, obj_vals, times

def log_barrier_objective(nu, X, y, group_matrices, t):
    reg_term = 0
    for i in range(len(group_matrices)):
        if np.linalg.norm(group_matrices[i] @ X.T @ nu) >= 1:
            return np.inf
        reg_term -= 1/t * np.log(1 - np.linalg.norm(group_matrices[i] @ X.T @ nu))
    return nu.T @ y + reg_term

def true_objective(nu, y):
    return nu.T @ y

def log_barrier_gradient(nu, X, y, group_matrices, t, precomputations):
    reg_term = 0
    for i in range(len(group_matrices)):
        denom = 1 - np.linalg.norm(nu.T @ X @ group_matrices[i].T) ** 2
        reg_term += 2/(t * denom) * (precomputations[i] @ nu)
    return y + reg_term

def log_barrier_hessian(nu, X, group_matrices, t, precomputations):
    hessian = 0
    for i in range(len(group_matrices)):
        denom = 1 - np.linalg.norm(nu.T @ X @ group_matrices[i].T) ** 2
        hessian += 2/(t * denom) * precomputations[i]
        nu_reshaped = nu.reshape((nu.shape[0], 1))
        hessian += 4/(t * denom ** 2) * precomputations[i] @ nu_reshaped @ nu_reshaped.T @ precomputations[i].T
    return hessian

def backtracking_line_search(x, dx, grad, f, a=0.25, b=0.5):
    t_line = 1
    while f(x + t_line * dx) > f(x) + a * t_line * np.dot(grad, dx):
        t_line = b * t_line
    return t_line

def newton(X, y, mu, t, group_matrices, precomputations, inner_eps=1e0, max_iter=50):
    start_time = time.time()
    n = y.shape[0]
    nu = np.zeros(n)
    obj_vec = np.zeros(max_iter)
    times = []
    for j in range(max_iter):
        nt_dec = np.inf
        while nt_dec / 2 >= inner_eps:
            grad = log_barrier_gradient(nu, X, y, group_matrices, t, precomputations)
            hess = log_barrier_hessian(nu, X, group_matrices, t, precomputations)
            x_nt = np.linalg.solve(hess, grad)
            x_nt = -x_nt
            nt_dec = np.dot(x_nt, hess @ x_nt)
            t_inner = backtracking_line_search(nu, x_nt, grad, partial (log_barrier_objective, t=t, X=X, y=y, group_matrices=group_matrices))
            nu += t_inner * x_nt
        obj_vec[j] = -true_objective(nu, y)
        times.append(time.time() - start_time)
        t *= mu
    return nu, obj_vec, times

def truncated_newton(X, y, mu, t, group_matrices, precomputations, cgiters, inner_eps=1e0, max_iter=50):
    start_time = time.time()
    n = y.shape[0]
    nu = np.zeros(n)
    obj_vec = np.zeros(max_iter)
    times = []
    for j in range(max_iter):
        nt_dec = np.inf
        while nt_dec / 2 >= inner_eps:
            grad = log_barrier_gradient(nu, X, y, group_matrices, t, precomputations)
            hess = log_barrier_hessian(nu, X, group_matrices, t, precomputations)
            x_nt, _ = cg(hess, grad, maxiter=cgiters)
            x_nt = -x_nt
            nt_dec = np.dot(x_nt, hess @ x_nt)
            t_inner = backtracking_line_search(nu, x_nt, grad, partial (log_barrier_objective, t=t, X=X, y=y, group_matrices=group_matrices))
            nu += t_inner * x_nt
        obj_vec[j] = -true_objective(nu, y)
        times.append(time.time() - start_time)
        t *= mu
    return nu, obj_vec, times

def truncated_newton_limited_hessian(X, y, mu, t, group_matrices, precomputations, inner_eps=1e0, max_iter=50):
    n = y.shape[0]
    nu = np.zeros(n)
    obj_vec = np.zeros(max_iter)

    for j in range(max_iter):
        nt_dec = np.inf
        counter = 0
        hess = None
        while nt_dec / 2 >= inner_eps:
            grad = log_barrier_gradient(nu, X, y, group_matrices, t, precomputations)
            if counter % 10 == 0:
                hess = log_barrier_hessian(nu, X, group_matrices, t, precomputations)
            x_nt, _ = cg(hess, grad, maxiter=20)
            x_nt = -x_nt
            nt_dec = np.dot(x_nt, hess @ x_nt)
            t_inner = backtracking_line_search(nu, x_nt, grad, partial (log_barrier_objective, t=t, X=X, y=y, group_matrices=group_matrices))
            nu += t_inner * x_nt
            counter += 1
        obj_vec[j] = -true_objective(nu, y)
        t *= mu
    return nu, obj_vec

def project_onto_ball(v, y, X, b, n):
    r_squared = 2 * n * b**2
    XTX = X.T @ X
    eps = 1e-6
    A = 0
    B = np.linalg.norm(v) ** 2
    lambd = 0
    while abs(B - A) > eps:
        lambd = (A+B) / 2
        inv = np.linalg.inv(np.eye(X.shape[1]) + lambd * XTX)
        g = 0.5 * v.T @ inv @ XTX @ inv @ v - r_squared/2
        if g > 0:
            A = (A+B)/2
        elif g < 0:
            B = (A+B)/2
        else:
            break
    u = np.linalg.inv(np.eye(X.shape[1]) + lambd * XTX) @ v
    return u + np.linalg.inv(X) @ y

def project_onto_ball_cvxpy(v, y, X, b, n):
    z = cp.Variable(v.shape)
    objective = cp.Minimize(cp.norm(z - v, 2) ** 2)
    constraints = [1/(2*n) * cp.norm(y - X @ z, 2) ** 2 <= b**2]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return z.value

def admm_low_dimensional(X, y, groups, rho, num_steps, alpha_init, beta_init, gamma_init, b):
    alpha = alpha_init
    beta = beta_init
    gamma = gamma_init
    obj_vals = np.zeros(num_steps)
    for k in range(num_steps):
        beta = group_lasso_prox(alpha - gamma, 1/rho, groups)
        alpha = project_onto_ball_cvxpy(beta + gamma, y, X, b, X.shape[0])
        gamma = gamma + beta - alpha
        obj_vals[k] = group_lasso_penalty(beta, groups)
    return beta, alpha, gamma, obj_vals

#BELOW CODE NOT WORKING YET

def wolfe_line_search(x, dx, grad_fn, f, c1 = 1e-4, c2 = 0.9, b = 0.5):
    t_line = 1
    grad = grad_fn(x)
    #while f(x + t_line * dx) > f(x) + a * t_line * np.dot(grad, dx):
    while f(x + t_line * dx) > f(x) + c1 * t_line * np.dot(grad, dx) or np.dot(dx, grad_fn(x + t_line * dx)) < c2 * np.dot(grad, dx):
        t_line = b * t_line
    # print('From line search:', f(x + t_line * dx) <= f(x) + 0.25 * t_line * np.dot(grad, dx)),→
    return t_line

def bfgs_newton(nu, X, y, group_matrices, precomputations, num_iters, num_newton_steps):
    t = 1
    mu = 1.5
    cur_sol = nu
    objective_vals = [log_barrier_objective(cur_sol, X, y, group_matrices, t)]
    for i in range(num_iters):
        print(i)
        H = np.eye(cur_sol.size)
        for j in range(num_newton_steps):
            grad = log_barrier_gradient(cur_sol, X, y, group_matrices, t, precomputations)
            #print(np.linalg.norm(grad))
            if np.linalg.norm(grad) <= 1e-5:
                break
            p = -H @ grad
            step = wolfe_line_search(cur_sol, p, partial(log_barrier_gradient, t=t, X=X, y=y, group_matrices=group_matrices, precomputations=precomputations), partial(log_barrier_objective, t=t, X=X, y=y, group_matrices=group_matrices))
            #print(step)
            s = step * p
            cur_sol = cur_sol + s
            y_bfgs = log_barrier_gradient(cur_sol, X, y, group_matrices, t, precomputations) - grad
            #H = H + ((s.T @ y + y.T @ H @ y) * np.outer(s, s))/((np.dot(s, y)) ** 2) - ((H @ np.outer(y, s) + np.outer(s, y) @ H)/(np.dot(s, y)))
            #rho = 1.0 / np.dot(y, s)
            #Hy = np.outer(H @ y, s)
            #H = H + rho * np.outer(s, s) - (rho * (Hy + Hy.T))
            #H = (np.eye(cur_sol.size) - (np.outer(s, y))/(np.dot(y, s))) @ H @ (np.eye(cur_sol.size) - (np.outer(y, s))/(np.dot(s, y))) + (np.outer(s, s))/(np.dot(y, s))
            #y = np.array([y])
            #s = np.array([s])
            y_bfgs = np.reshape(y_bfgs, (y_bfgs.shape[0], 1))
            s = np.reshape(s, (s.shape[0], 1))
            r = 1/(y_bfgs.T @ s)
            li = (np.eye(y_bfgs.shape[0]) - (r*((s@(y_bfgs.T)))))
            ri = (np.eye(s.shape[0]) -(r*((y_bfgs@(s.T)))))
            hess_inter = li@H@ri
            #H = H + ((np.dot(s, y) + y.T @ H @ y) * np.outer(s, s))/(np.dot(s, y) ** 2) - (H @ np.outer(y, s) + np.outer(s, y) @ H)/(np.dot(s, y))
            H = hess_inter + (r*((s@(s.T))))
            #print(H.shape)
            #print(cur_sol)
            #print(-log_barrier_objective(cur_sol, X, y, group_matrices, t))
        print(-log_barrier_objective(cur_sol, X, y, group_matrices, t))
        print(cur_sol)
        objective_vals.append(-log_barrier_objective(cur_sol, X, y, group_matrices, t))
        t *= mu
    return objective_vals, cur_sol
