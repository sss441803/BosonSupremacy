import numpy as np
import cupy as cp
from tqdm import tqdm
from scipy.linalg import sqrtm, svd, block_diag, schur
from math import ceil
import time

def nothing_function(object):
    return object

tqdm = nothing_function

complex_type = 'complex64'

kernel_file = open('direct_mps_kernels.cu')
kernel_string = kernel_file.read()
kernel_file.close()
sigma_select_cfloat = cp.RawKernel(kernel_string, 'sigma_select_cfloat')
sigma_select_cdouble = cp.RawKernel(kernel_string, 'sigma_select_cdouble')

def Sigma_select(Sigma, target):
    max_blocks = 65535
    n_batch, n_select = target.shape
    n_len = Sigma.shape[0]
    target = cp.array(target, dtype='int32')
    Sigma = cp.array(Sigma, dtype=complex_type)
    Sigma2 = cp.zeros([n_batch, n_select, n_select], dtype=complex_type)
    threadsperblock = (4, 4, 16)
    blockspergrid = (ceil(n_select/4), ceil(n_select/4), ceil(n_batch/16))
    if complex_type == 'complex64':
        sigma_select = sigma_select_cfloat
    elif complex_type == 'complex128':
        sigma_select = sigma_select_cdouble
    else:
        raise ValueError
    for superblock_id in range(blockspergrid[2] // max_blocks + 1):
        begin_block = superblock_id * max_blocks
        end_block = min((superblock_id + 1) * max_blocks, blockspergrid[2])
        begin_batch = begin_block * 16
        end_batch = min(n_batch, end_block * 16)
        launch_n_batch = end_batch - begin_batch
        launch_blockspergrid = (blockspergrid[0], blockspergrid[1], end_block - begin_block)
        launch_idx = target[begin_batch : end_batch]
        launch_Sigma2 = Sigma2[begin_batch : end_batch]
        # print(launch_idx, Sigma, launch_blockspergrid, threadsperblock, begin_batch, end_batch)
        sigma_select(launch_blockspergrid, threadsperblock, (launch_n_batch, n_select, n_len, Sigma, launch_idx, launch_Sigma2))
    return Sigma2
    
def push_to_end(array):
    n_batch, n_select = array.shape
    new_array = np.zeros_like(array)
    idx = np.zeros(n_batch, dtype='int32')
    for i in range(1, n_select + 1):
        occupied = array[:, -i] != 0
        idx += occupied
        new_array[np.arange(n_batch), -idx] = array[:, -i]
    return new_array

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

def hafnian(A): 

    matshape = A.shape[1:]
    n_batch = A.shape[0]

    if matshape == (0, 0):
        return cp.ones(n_batch, dtype='complex64')
    
    if matshape[0] % 2 != 0:
        return cp.zeros(n_batch, dtype='complex64')
    
    '''removed case where it is identity'''
    if matshape[0] == 2:
        return A[:, 0, 1]

    if matshape[0] == 3:
        return (
            A[:, 0, 0] * A[:, 1, 2] + A[:, 1, 1] * A[:, 0, 2] + A[:, 2, 2] * A[:, 0, 1] + A[:, 0, 0] * A[:, 1, 1] * A[:, 2, 2]
        )

    if matshape[0] == 4:
        return A[:, 0, 1] * A[:, 2, 3] + A[:, 0, 2] * A[:, 1, 3] + A[:, 0, 3] * A[:, 1, 2]

    return recursive_hafnian(A)


def recursive_hafnian(A):  # pragma: no cover

    n_batch, nb_lines, nb_columns = A.shape
    if nb_lines != nb_columns:
        raise ValueError("Matrix must be square")

    if nb_lines % 2 != 0:
        raise ValueError("Matrix size must be even")

    n = A.shape[1] // 2
    z = cp.zeros((n_batch, n * (2 * n - 1), n + 1), dtype=A.dtype)
    for j in range(1, 2 * n):
        ind = j * (j - 1) // 2
        for k in range(j):
            z[:, ind + k, 0] = A[:, j, k]
    g = cp.zeros([n_batch, n + 1], dtype=A.dtype)
    g[:, 0] = 1
    return solve(z, 2 * n, 1, g, n)


def solve(b, s, w, g, n):  # pragma: no cover

    n_batch = b.shape[0]
    if s == 0:
        return w * g[:, n]
    c = cp.zeros((n_batch, (s - 2) * (s - 3) // 2, n + 1), dtype=g.dtype)
    i = 0
    for j in range(1, s - 2):
        for k in range(j):
            c[:, i] = b[:, (j + 1) * (j + 2) // 2 + k + 2]
            i += 1
    h = solve(c, s - 2, -w, g, n)
    e = g.copy()
    for u in range(n):
        for v in range(n - u):
            e[:, u + v + 1] += g[:, u] * b[:, 0, v]
    for j in range(1, s - 2):
        for k in range(j):
            for u in range(n):
                for v in range(n - u):
                    c[:, j * (j - 1) // 2 + k, u + v + 1] += (
                        b[:, (j + 1) * (j + 2) // 2, u] * b[:, (k + 1) * (k + 2) // 2 + 1, v]
                        + b[:, (k + 1) * (k + 2) // 2, u] * b[:, (j + 1) * (j + 2) // 2 + 1, v]
                    )
    return h + solve(c, s - 2, w, e, n)

def blochmessiah(S):
    N, _ = S.shape

    # Changing Basis
    R = (1 / np.sqrt(2)) * np.block(
        [[np.eye(N // 2), 1j * np.eye(N // 2)], [np.eye(N // 2), -1j * np.eye(N // 2)]]
    )
    Sc = R @ S @ np.conjugate(R).T
    # Polar Decomposition
    # u1, d1, v1 = np.linalg.svd(Sc)
    u1, d1, v1 = svd(Sc, lapack_driver='gesvd')
    Sig = u1 @ np.diag(d1) @ np.conjugate(u1).T
    Unitary = u1 @ v1
    # Blocks of Unitary and Hermitian symplectics
    alpha = Unitary[0 : N // 2, 0 : N // 2]
    beta = Sig[0 : N // 2, N // 2 : N]
    # Bloch-Messiah in this Basis
    u2, d2, v2 = np.linalg.svd(beta)
    sval = np.arcsinh(d2)
    takagibeta = u2 @ sqrtm(np.conjugate(u2).T @ (v2.T))
    uf = np.block([[takagibeta, 0 * takagibeta], [0 * takagibeta, np.conjugate(takagibeta)]])
    vf = np.block(
        [
            [np.conjugate(takagibeta).T @ alpha, 0 * takagibeta],
            [0 * takagibeta, np.conjugate(np.conjugate(takagibeta).T @ alpha)],
        ]
    )
    df = np.block(
        [
            [np.diag(np.cosh(sval)), np.diag(np.sinh(sval))],
            [np.diag(np.sinh(sval)), np.diag(np.cosh(sval))],
        ]
    )
    # Rotating Back to Original Basis
    uff = np.conjugate(R).T @ uf @ R
    vff = np.conjugate(R).T @ vf @ R
    dff = np.conjugate(R).T @ df @ R
    dff = np.real_if_close(dff)
    vff = np.real_if_close(vff)
    uff = np.real_if_close(uff)
    return uff, dff, vff

def thermal_photons(nth, cutoff = 20):
    return 1 / (nth + 1) * (nth / (nth + 1)) ** np.arange(cutoff)

def get_Sigma(U2, sq, U1):
    M = len(sq)
    Sigma = np.zeros((2 * M, 2 * M), dtype=complex_type)
    Sigma[:M, :M] = U2 @ np.diag(np.tanh(sq)) @ U2.T
    Sigma[:M, M:] = U2 @ np.diag(1 / np.cosh(sq)) @ U1
    Sigma[M:, :M] = U1.T @ np.diag(1 / np.cosh(sq)) @ U2.T
    Sigma[M:, M:] = -U1.T @ np.diag(np.tanh(sq)) @ U1
    return Sigma.astype(complex_type)

def get_target(num):
    n_select = np.sum(num, axis=1).max()
    n_batch, n_len = num.shape
    idx_x = np.tile(np.arange(n_select, dtype='int32').reshape(1, -1), (n_batch, 1))
    target = np.zeros([n_batch, n_select], dtype='int32')
    idx_end = np.zeros(n_batch)
    idx_begin = np.zeros(n_batch)
    num = np.array(num)
    for i in range(n_len):
        vals_n = num[:, i]
        # vals_n = cp.array(n_[:, i])
        idx_end += vals_n
        # print(idx_begin, idx_end)
        mask = idx_x >= idx_begin.reshape(-1, 1)
        mask *= idx_x < idx_end.reshape(-1, 1)
        # print(mask)
        target += mask * (i + 1)
        idx_begin = np.copy(idx_end)
    return cp.array(target, dtype='int32')

def A_elem(Sigma, target, denominator, max_memory_in_gb):
    # print(target.shape)
    n_batch, n_select = target.shape
    all_haf = cp.zeros([0], dtype='complex64')
    if n_select == 0:
        n_batch_max = 99999999999
    else:
        n_batch_max = int(max_memory_in_gb * (10 ** 9) // (n_select ** 2 * 8))
    # print(n_batch_max)
    sigma_time = 0
    haf_time = 0
    for begin_batch in tqdm(range(0, n_batch, n_batch_max)):
        end_batch = min(n_batch, begin_batch + n_batch_max)
        start = time.time()
        Sigma2 = Sigma_select(Sigma, target[begin_batch : end_batch])
        cp.cuda.runtime.deviceSynchronize()
        sigma_time += time.time() - start
        start = time.time()
        haf = hafnian(Sigma2).astype('complex64')
        # haf = cp.zeros([Sigma2.shape[0]], dtype='complex64')
        cp.cuda.runtime.deviceSynchronize()
        haf_time += time.time() - start
        # print(haf)
        all_haf = cp.append(all_haf, haf)
    return all_haf / denominator, haf_time, sigma_time

def get_U2_sq_U1(S_l, S_r):
    M = len(S_r) // 2
    mode = np.arange(M - 1) + 1
    modes = np.append(mode, mode + M)
    
    S_l2_inv = np.eye(2 * M, dtype = float)
    S_l2_inv[np.ix_(modes, modes)] = np.linalg.inv(S_l)
    S = S_l2_inv @ S_r
    
    S2, SQ, S1 = blochmessiah(S)
    U2 = S2[:M, :M] - 1j * S2[:M, M:]
    U1 = S1[:M, :M] - 1j * S1[:M, M:]

    sq = np.log(np.diag(SQ)[:M])
    
    return U2, sq, U1