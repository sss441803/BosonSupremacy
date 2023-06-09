import numpy as np
from scipy.linalg import block_diag, sqrtm, schur
import argparse
import os

def nothing_function(object):
    return object

tqdm = nothing_function

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--dir', type=str, help="Root directory.", default=0)
args = vars(parser.parse_args())

d = args['d']
chi = args['chi']
rootdir = args['dir']



def sympmat(N, dtype=np.float64):
    I = np.identity(N, dtype=dtype)
    O = np.zeros_like(I, dtype=dtype)
    S = np.block([[O, I], [-I, O]])
    return S

def xpxp_to_xxpp(S):
    shape = S.shape
    n = shape[0]

    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(-1, 2).T.flatten()

    if len(shape) == 2:
        if shape[0] != shape[1]:
            raise ValueError("The input matrix is not square")
        return S[:, ind][ind]

    return S[ind]

def williamson(V, tol=1e-11):
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    diffn = np.linalg.norm(V - np.transpose(V))

    if diffn >= tol:
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = sympmat(n)
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = xpxp_to_xxpp(s1t)
    perm_indices = xpxp_to_xxpp(np.arange(2 * n))
    Ktt = Kt[:, perm_indices]
    Db = np.diag([1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T


def thermal_photons(nth, cutoff = 20):
    return 1 / (nth + 1) * (nth / (nth + 1)) ** np.arange(cutoff)

def get_cumsum_kron(sq_cov, L, chi = 100, max_dim = 10 ** 5, cutoff = 6, err_tol = 10 ** (-12)):
    M = len(sq_cov) // 2
    mode = np.arange(L, M)
    modes = np.append(mode, mode + M)
    sq_cov_A = sq_cov[np.ix_(modes, modes)]

    D, S = williamson(sq_cov_A)
    d = (np.diag(D) - 1) / 2

    d[d < 0] = 0

    res = thermal_photons(d[0], cutoff)
    num = np.arange(cutoff, dtype='int8')
    
    for i in range(1, M - L):
        res = np.outer(res, np.array(thermal_photons(d[i], cutoff))).reshape(-1)
        keep_idx = np.where(res > err_tol)[0]
        res = res[keep_idx]
        idx = np.argsort(res)[-min(len(res), max_dim):]       
        res = res[idx][::-1]
        '''Instead of creating the full cartesian product, use the keep_idx variable to reduce the amount of data we need to generate and write a custom cuda kernel'''
        if len(num.shape) == 1:
            num = num.reshape(-1, 1)
        keep_idx = keep_idx[idx][::-1]
        num = np.concatenate([num[keep_idx // cutoff], np.arange(cutoff).reshape(-1, 1)[keep_idx % cutoff]], axis=1)
            
    len_ = min(chi, len(res))
    idx = np.argsort(res)[-len_:]
    idx_sorted = idx[np.argsort(res[idx])]
    res = res[idx_sorted][::-1]
    num = num[idx_sorted][::-1]

    return res.astype('float16'), num.astype('int8'), S




if __name__ == "__main__":

    path = rootdir + f"d_{d}_chi_{chi}/"
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    M = len(cov) // 2

    if not os.path.isdir(path):
        os.mkdir(path)
    active_sites = np.zeros(M - 1, dtype='int32')
    np.save(path + 'active_kron_sites.npy', active_sites)

    max_memory_in_gb = 1
    max_dim = 10 ** 5

    for compute_site in range(M - 1):
        
        res, num, S_l = get_cumsum_kron(sq_cov, compute_site + 1, max_dim = max_dim, chi = chi, cutoff = d)
        print(compute_site, np.sum(res))
        np.save(path + f'res_{compute_site}.npy', res)
        np.save(path + f'num_{compute_site}.npy', num)
        np.save(path + f'S_{compute_site}.npy', S_l)