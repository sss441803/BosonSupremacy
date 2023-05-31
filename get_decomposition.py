import numpy as np
import cvxpy as cp
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help="Experiment directory.")
args = vars(parser.parse_args())

dir = args['dir']

if os.path.isfile(dir + 'sq_cov.npy'):
    print('Decomposition results already available.')
    quit()

cov = np.load(dir + "cov.npy")
omega = np.array([[0, 1], [-1, 0]])
M = len(cov) // 2;
Omega = np.kron(omega, np.eye(M))

n = 2 * M
X = cp.Variable((n,n), symmetric=True)
constraints = [cp.bmat([[X, Omega], [-Omega, X]]) >> 0]
constraints += [cov - X >> 0]
prob = cp.Problem(cp.Minimize(cp.trace(X)),
                    constraints)
prob.solve(solver = 'CVXOPT')
sq_cov = X.value

np.save(dir + 'sq_cov.npy', sq_cov)