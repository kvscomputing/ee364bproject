import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from utils import generate_data, generate_group_matrices, precompute_log_barrier_terms
from methods import admm, newton, truncated_newton, truncated_newton_limited_hessian, group_lasso_penalty, bfgs_newton, admm_low_dimensional
import time

np.random.seed(1734)

n = 200
p = 500

X, y, groups = generate_data(n=n, p=p, rho=0.5, group_size=5)
#print(X)
#print(X, y, groups)
group_matrices = generate_group_matrices(groups)
#for i in range(len(groups)):
    #print(group_matrices[i] @ X.T)
log_barrier_terms = precompute_log_barrier_terms(X, group_matrices)

beta = cp.Variable(p)
group_norms = [cp.norm(beta[group], 2) for group in groups]
objective = cp.Minimize(cp.sum(group_norms))
constraints = [y == X @ beta]

problem = cp.Problem(objective, constraints)
problem.solve()

fstar = objective.value
print(fstar)

alpha_init = np.zeros(p)
beta_init = np.zeros(p)
gamma_init = np.zeros(p)
for i in range(p):
    alpha_init[i] += 1e-3
    beta_init[i] += 1e-3
    gamma_init[i] += 1e-1
num_steps = 500
rho = 0.1

rho_values = [0.1, 0.5, 1.0, 5.0, 10.0]

admm_obj_vals_results = {}

'''
for rho in rho_values:
    beta, alpha, gamma, admm_obj_vals = admm(X, y, groups, rho, num_steps, alpha_init, beta_init, gamma_init)
    admm_obj_vals_results[rho] = admm_obj_vals
    print("HERE")
    print(admm_obj_vals[-1])

for rho, admm_obj_vals in admm_obj_vals_results.items():
    plt.plot([abs(val - fstar) for val in admm_obj_vals], label=f"rho = {rho}")

plt.ylabel("Optimality Gap")
plt.xlabel("Iterations")
plt.legend()
plt.yscale("log")
plt.title("ADMM Convergence")
plt.savefig("admm_rho.png")
'''

start_time = time.time()
beta, alpha, gamma, admm_obj_vals, admm_times = admm(X, y, groups, rho, num_steps, alpha_init, beta_init, gamma_init)
#print(admm_obj_vals)
print(group_lasso_penalty(beta, groups))
print(abs(start_time - time.time()))

nu_init = np.zeros(n)
precomputations = precompute_log_barrier_terms(X, group_matrices)
num_iters = 50
num_newton_steps = 100

nu = cp.Variable(n)
objective = cp.Minimize(nu.T @ y)
constraints = [cp.norm(group_matrices[g] @ X.T @ nu, 2) <= 1 for g in range(len(group_matrices))]

problem = cp.Problem(objective, constraints)
problem.solve()
print(-objective.value)
#print(nu.value)

start_time = time.time()
nu, newton_obj_vec, newton_times = newton(X, y, 1.5, 1.0, group_matrices, precomputations)
print(-nu.T @ y)
print(abs(start_time - time.time()))

cgiter_values = [1, 5, 10, 20, 30, 50]

truncated_newton_results = {}
'''
for cgiter_val in cgiter_values:
    print(cgiter_val)
    start_time = time.time()
    nu, truncated_newton_obj_vec = truncated_newton(X, y, 1.5, 1.0, group_matrices, precomputations, cgiter_val)
    print(abs(start_time - time.time()))
    truncated_newton_results[cgiter_val] = truncated_newton_obj_vec
plt.clf()
for cgiter_val, truncated_newton_obj_vec in truncated_newton_results.items():
    plt.plot([abs(val - fstar) for val in truncated_newton_obj_vec], label=f"cgiters = {cgiter_val}")

plt.ylabel("Optimality Gap")
plt.xlabel("Iterations")
plt.legend()
plt.yscale("log")
plt.title("Truncated Newton Convergence")
plt.savefig("cgiter.png")
'''
start_time = time.time()
nu, truncated_newton_obj_vec, truncated_newton_times = truncated_newton(X, y, 1.5, 1.0, group_matrices, precomputations, 20)
#print(nu)
print(-nu.T @ y)
print(abs(start_time - time.time()))

start_time = time.time()
nu, limited_hessian_obj_vec = truncated_newton_limited_hessian(X, y, 1.5, 1.0, group_matrices, precomputations)
print(-nu.T @ y)
print(abs(start_time - time.time()))

plt.clf()
plt.plot(admm_times, [abs(val - fstar) for val in admm_obj_vals], label="ADMM")
plt.plot(newton_times, [abs(val - fstar) for val in newton_obj_vec], label="Newton")
plt.plot(truncated_newton_times, [abs(val - fstar) for val in truncated_newton_obj_vec], label="Truncated Newton")
plt.ylabel("Optimality Gap")
plt.xlabel("Times")
plt.legend()
plt.yscale("log")
plt.title("Convergence Time Across Different Methods")
plt.savefig("times.png")

#plt.plot([val - fstar for val in admm_obj_vals], label="ADMM")
plt.clf()
plt.plot([abs(fstar - val) for val in newton_obj_vec], label="Newton")
plt.plot([abs(fstar - val) for val in truncated_newton_obj_vec], label="Truncated Newton")
plt.plot([abs(fstar - val) for val in limited_hessian_obj_vec], label="Limited Hessian Truncated Newton")
plt.ylabel("Optimality Gap")
plt.xlabel("Iterations")
plt.legend()
plt.yscale("log")
plt.title("Comparing Newton Methods")
plt.savefig("newtonplt.png")

plt.clf()
plt.plot([abs(val - fstar) for val in admm_obj_vals], label="ADMM")
plt.ylabel("Optimality Gap")
plt.xlabel("Iterations")
plt.legend()
plt.yscale("log")
plt.title("ADMM Convergence")
plt.savefig("admm.png")


'''
#objective_vals, cur_sol = bfgs_newton(nu_init, X, y, group_matrices, precomputations, num_iters, num_newton_steps)
#print(cur_sol)
#print(-cur_sol.T @ y)


n = 200
p = 100

X, y, groups = generate_data(n=n, p=p, rho=2.0, group_size=2)

alpha_init = np.zeros(p)
beta_init = np.zeros(p)
gamma_init = np.zeros(p)
for i in range(p):
    alpha_init[i] += 1e-3
    beta_init[i] += 1e-3
    gamma_init[i] += 1e-2
num_steps = 1000
rho = 0.5

b = 10

beta = cp.Variable(p)
group_norms = [cp.norm(beta[group], 2) for group in groups]
objective = cp.Minimize(cp.sum(group_norms))
constraints = [1/(2 * X.shape[0]) * cp.norm(X @ beta - y, 2)**2 <= b**2]

problem = cp.Problem(objective, constraints)
problem.solve()

print(objective.value)

start_time = time.time()
beta, alpha, gamma, admm_obj_vals = admm_low_dimensional(X, y, groups, rho, num_steps, alpha_init, beta_init, gamma_init, b=b)
print(group_lasso_penalty(beta, groups))
print(abs(start_time - time.time()))
'''