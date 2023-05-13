import numpy as np
import cupy as cp
from scipy.linalg import block_diag, sqrtm, schur
import argparse
import time
import os
from filelock import FileLock
from mpi4py import MPI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
args = vars(parser.parse_args())

d = args['d']
chi = args['chi']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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


def get_cumsum_kron1(sq_cov, L, chi = 100, max_dim = 10 ** 5, cutoff = 6, err_tol = 10 ** (-12)):
    M = len(sq_cov) // 2
    mode = np.arange(L, M)
    modes = np.append(mode, mode + M)
    sq_cov_A = sq_cov[np.ix_(modes, modes)]

    D, S = williamson(sq_cov_A)
    d = (np.diag(D) - 1) / 2

    d[d < 0] = 0

    res = thermal_photons(d[0], cutoff)
    num = np.arange(cutoff, dtype='int8')
    
    kron_time = 0
    cart_time = 0
    select_time = 0
    sort_time = 0
    rev_time = 0
    
    for i in tqdm(range(1, M - L)):
        start = time.time()
        res = np.kron(res, thermal_photons(d[i], cutoff))
        kron_time += time.time() - start
        start = time.time()
        keep_idx = np.where(res > err_tol)[0]
        start = time.time()
        '''Instead of creating the full cartesian product, use the keep_idx variable to reduce the amount of data we need to generate and write a custom cuda kernel'''
        # orig_num = cartesian(num, np.arange(cutoff))
        # orig_num = orig_num[keep_idx]
        # print(keep_idx, num)
        if len(num.shape) == 1:
            num = num.reshape(-1, 1)
        num = np.concatenate([num[keep_idx // cutoff], np.arange(cutoff).reshape(-1, 1)[keep_idx % cutoff]], axis=1)
        # print('compare ', orig_num, num)
        cart_time += time.time() - start
        res = res[keep_idx]
        select_time += time.time() - start
        start = time.time()
        idx = np.argsort(res)[-min(len(res), max_dim):]       
        #if len(res) > max_dim:
        sort_time += time.time() - start
        start = time.time()
        res = res[idx][::-1]
        num = num[idx][::-1]
        rev_time += time.time() - start

    print('loop time ', kron_time, cart_time, select_time, sort_time, rev_time)
            
    len_ = min(chi, len(res))
    idx = np.argsort(res)[-len_:]
    idx_sorted = idx[np.argsort(res[idx])]
    res = res[idx_sorted][::-1]
    num = num[idx_sorted][::-1]
    
    print(res.shape, num.shape)

    return res, num, S


def get_cumsum_kron(sq_cov, L, chi = 100, max_dim = 10 ** 5, cutoff = 6, err_tol = 10 ** (-12)):
    M = len(sq_cov) // 2
    mode = np.arange(L, M)
    modes = np.append(mode, mode + M)
    sq_cov_A = sq_cov[np.ix_(modes, modes)]

    D, S = williamson(sq_cov_A)
    d = (np.diag(D) - 1) / 2

    d[d < 0] = 0

    res = cp.array(thermal_photons(d[0], cutoff))
    num = cp.arange(cutoff, dtype='int8')
    
    kron_time = 0
    cart_time = 0
    select_time = 0
    sort_time = 0
    rev_time = 0
    
    for i in tqdm(range(1, M - L)):
        start = time.time()
        # orig_res = cp.kron(res, cp.array(thermal_photons(d[i], cutoff)))
        # new_res = cp.outer(res, cp.array(thermal_photons(d[i], cutoff))).reshape(-1)
        # assert cp.allclose(orig_res, new_res)
        # res = orig_res
        res = cp.outer(res, cp.array(thermal_photons(d[i], cutoff))).reshape(-1)
        kron_time += time.time() - start
        start = time.time()
        keep_idx = cp.where(res > err_tol)[0]
        start = time.time()
        '''Instead of creating the full cartesian product, use the keep_idx variable to reduce the amount of data we need to generate and write a custom cuda kernel'''
        # orig_num = cartesian(num, np.arange(cutoff))
        # orig_num = orig_num[keep_idx]
        # print(keep_idx, num)
        if len(num.shape) == 1:
            num = num.reshape(-1, 1)
        num = cp.concatenate([num[keep_idx // cutoff], cp.arange(cutoff).reshape(-1, 1)[keep_idx % cutoff]], axis=1)
        # print('compare ', orig_num, num)
        cart_time += time.time() - start
        res = res[keep_idx]
        select_time += time.time() - start
        start = time.time()
        idx = cp.argsort(res)[-min(len(res), max_dim):]       
        #if len(res) > max_dim:
        sort_time += time.time() - start
        start = time.time()
        res = res[idx][::-1]
        num = num[idx][::-1]
        rev_time += time.time() - start

    print('Time: kron {}, cart {}, select {}, sort {}, rev {}.'.format(kron_time, cart_time, select_time, sort_time, rev_time))
            
    len_ = min(chi, len(res))
    idx = cp.argsort(res)[-len_:]
    idx_sorted = idx[np.argsort(res[idx])]
    res = res[idx_sorted][::-1]
    num = num[idx_sorted][::-1]
    
    print(res.shape, num.shape)

    return cp.asnumpy(res), cp.asnumpy(num), S




if __name__ == "__main__":

    rootdir = "/project2/liangjiang/mliu6/DirectMPS/data_S15/"
    path = rootdir + f"d_{d}_chi_{chi}/"
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    sq_array = np.load(rootdir + "sq_array.npy")
    M = len(cov) // 2

    if not os.path.isfile(path + 'active_kron_sites.npy'):
        if rank == 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            active_sites = np.zeros(M - 1, dtype='int32')
            np.save(path + 'active_kron_sites.npy', active_sites)

        completed = True
        comm.bcast(completed, root=0)

    max_memory_in_gb = 1
    max_dim = 10 ** 5

    while True:
        with FileLock(path + 'active_kron_sites.npy.lock'):
            print(f'Rank {rank} acquired lock.')
            active_sites = np.load(path + 'active_kron_sites.npy')
            # print(active_sites)
            uncomputed_sites = np.where(active_sites == 0)[0]
            if uncomputed_sites.shape[0] == 0:
                print(f'Rank {rank} all completed.')
                quit()
            compute_site = uncomputed_sites[0]
            active_sites[compute_site] = 1
            np.save(path + 'active_kron_sites.npy', active_sites)

        res, num, S_l = get_cumsum_kron(sq_cov, compute_site + 1, max_dim = max_dim, chi = chi, cutoff = d)
        np.save(path + f'res_{compute_site}.npy', res)
        np.save(path + f'num_{compute_site}.npy', num)
        np.save(path + f'S_{compute_site}.npy', S_l)