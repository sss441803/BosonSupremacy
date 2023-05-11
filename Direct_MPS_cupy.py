import numpy as np
import cupy as cp
from scipy.special import factorial
from scipy.linalg import sqrtm, svd
from math import ceil
import time
from thewalrus import hafnian as default_hafnian
from thewalrus.symplectic import is_symplectic
from strawberryfields.decompositions import williamson

kernel_file = open('direct_mps_kernels.cu')
kernel_string = kernel_file.read()
kernel_file.close()
sigma_select = cp.RawKernel(kernel_string, 'sigma_select')

def Sigma_select(Sigma, target):
    max_blocks = 65535
    n_batch, n_select = target.shape
    n_len = Sigma.shape[0]
    target = cp.array(target, dtype='int32')
    Sigma = cp.array(Sigma, dtype='complex128')
    Sigma2 = cp.zeros([n_batch, n_select, n_select], dtype='complex128')
    threadsperblock = (4, 4, 16)
    blockspergrid = (ceil(n_select/4), ceil(n_select/4), ceil(n_batch/16))
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
    # n_batch, nb_lines, nb_columns = A.shape
    # if nb_lines != nb_columns:
    #     raise ValueError("Matrix must be square")

    # if nb_lines % 2 != 0:
    #     raise ValueError("Matrix size must be even")

    # n = A.shape[1] // 2
    # A = cp.asnumpy(A)
    # z = np.zeros((n_batch, n * (2 * n - 1), n + 1), dtype=A.dtype)
    # for j in range(1, 2 * n):
    #     ind = j * (j - 1) // 2
    #     for k in range(j):
    #         z[:, ind + k, 0] = A[:, j, k]
    # g = np.zeros([n_batch, n + 1], dtype=A.dtype)
    # g[:, 0] = 1
    # return cp.array(solve(z, 2 * n, 1, g, n))


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

    if not is_symplectic(S):
        raise ValueError("Input matrix is not symplectic.")

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

def cartesian(array1, array2):
    # array1_orig = array1
    # array2_orig = array2
    array1 = np.array(array1)
    array2 = np.array(array2)
    if len(array1.shape) == 1:
        array1 = array1.reshape(-1, 1)
    if len(array2.shape) == 1:
        array2 = array2.reshape(-1, 1)
    len1 = array1.shape[0]
    len2 = array2.shape[0]
    array1 = np.repeat(array1, len2, 0)
    array2 = np.tile(array2, (len1, 1))
    # assert np.allclose(np.concatenate([array1, array2], axis=1), np.array([np.append(a, b) for a in array1_orig for b in array2_orig]))
    return np.concatenate([array1, array2], axis=1)

    # return np.array([np.append(a, b) for a in array1 for b in array2])


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
    
    kron_time = 0
    cart_time = 0
    select_time = 0
    sort_time = 0
    rev_time = 0
    
    for i in range(1, M - L):
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
    
    # res = cp.asnumpy(res)
    # num = cp.asnumpy(num)
    print(res.shape, num.shape)

    return res, num, S

def get_Sigma(U2, sq, U1):
    M = len(sq)
    Sigma = np.zeros((2 * M, 2 * M), dtype = complex)
    Sigma[:M, :M] = U2 @ np.diag(np.tanh(sq)) @ U2.T
    Sigma[:M, M:] = U2 @ np.diag(1 / np.cosh(sq)) @ U1
    Sigma[M:, :M] = U1.T @ np.diag(1 / np.cosh(sq)) @ U2.T
    Sigma[M:, M:] = -U1.T @ np.diag(np.tanh(sq)) @ U1
    return Sigma.astype('complex128')

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
    start = time.time()
    Sigma2 = Sigma_select(Sigma, target)
    sigma_time = time.time() - start
    start = time.time()
    haf = hafnian(Sigma2).astype('complex64')
    # haf = cp.zeros([Sigma2.shape[0]], dtype='complex64')
    cp.cuda.runtime.deviceSynchronize()
    # print(haf)
    return haf / denominator, time.time() - start, sigma_time

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


rootdir = "./data_MPS/data_3a/"
sq_cov = np.load(rootdir + "sq_cov.npy")
cov = np.load(rootdir + "cov.npy")
sq_array = np.load(rootdir + "sq_array.npy")
M = len(cov) // 2

real_start = time.time()

d = 4
chi = 1000
max_memory_in_gb = 1

Gamma = np.zeros((chi, chi, d, M), dtype = 'complex64')
Lambda = cp.zeros((chi, M - 1), dtype = 'float32')

max_dim = 10 ** 5; err_tol = 10 ** (-10)


i = 0
_, S_r = williamson(sq_cov)
res, num, S_l = get_cumsum_kron(sq_cov, i + 1, max_dim = max_dim, chi = chi, cutoff = d)
num = num[res > err_tol]
res = res[res > err_tol]
print(np.sum(res))
# obtain the singular values and eigen state of the right hand side.

U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
Sigma = get_Sigma(U2, sq, U1)

Z = np.sqrt(np.prod(np.cosh(sq)))

tot_haf_time = 0
tot_a_elem_time = 0
tot_gamma_time = 0
tot_sigma_time = 0
tot_target_time = 0
tot_kron_time = 0
tot_prep_time = 0
largest_sigma2_dim = 0
left_target = get_target(num)
left_n_select = left_target.shape[1]
left_sum = np.sum(num, axis=1)
left_denominator = cp.sqrt(cp.product(cp.array(factorial(num)), axis=1))
Lambda[:len(res), 0] = cp.array(np.sqrt(res))
for j in np.arange(d):
    for size in np.arange(np.max(left_sum) + 1):
        left_idx = np.where(left_sum == size)[0]
        if (Lambda[left_idx, i] <= err_tol).all():
            continue
        n_batch = left_idx.shape[0]
        '''one is already added to the left charge in function get_target'''
        target = cp.append(cp.zeros([n_batch, j], dtype='int32'), left_target[:, :size][left_idx], axis=1)
        denominator = cp.sqrt(factorial(j)) * left_denominator[left_idx]
        start = time.time()
        haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
        tot_a_elem_time += time.time() - start
        tot_haf_time += haf_time
        tot_sigma_time += sigma_time
        Gamma[0, left_idx, j, i] = cp.asnumpy(haf / Z / Lambda[left_idx, i])

S_r = np.copy(S_l)

res_pre = np.copy(res)
num_pre = np.copy(num)
right_target = cp.array(push_to_end(cp.asnumpy(left_target)))
right_sum = cp.array(np.sum(num, axis=1))
right_denominator = left_denominator
# print(num.shape, num)

for i in range(1, M - 1):
    start = time.time()
    res, num, S_l = get_cumsum_kron(sq_cov, i + 1, max_dim = max_dim, chi = chi, cutoff = d)
    # cp.cuda.runtime.deviceSynchronize()
    tot_kron_time += time.time() - start
    start = time.time()
    num = num[res > err_tol]
    num = num.reshape(num.shape[0], -1)
    left_target = get_target(num)
    left_n_select = left_target.shape[1]
    left_sum = cp.array(np.sum(num, axis=1))
    full_sum = cp.repeat(left_sum.reshape(-1, 1), right_sum.shape[0], axis=1) + cp.repeat(right_sum.reshape(1, -1), left_sum.shape[0], axis=0)
    left_denominator = cp.sqrt(cp.product(cp.array(factorial(num)), axis=1))
    print(i, np.sum(num, axis=1).max())
    res = res[res > err_tol]    
    print(np.sum(res))
    U2, sq, U1 = get_U2_sq_U1(S_l, S_r) # S_l: left in equation, S_r : right in equation
    Sigma = get_Sigma(U2, sq, U1)
    Z = np.sqrt(np.prod(np.cosh(sq)))
    Lambda[:len(res), i] = cp.array(np.sqrt(res))
    tot_prep_time += time.time() - start
    for j in np.arange(d):
        gpu_Gamma = cp.zeros([chi, chi], dtype='complex64')
        for size in np.arange(int(cp.nanmax(full_sum)) + 1):
            # cp.get_default_memory_pool().free_all_blocks()
            start = time.time()
            left_idx, right_idx = cp.where(full_sum == size)
            n_batch = left_idx.shape[0]
            if (Lambda[left_idx, i] <= err_tol).all():
                continue
            if size == 0 and j == 0:
                # print('r, l ', right_idx, left_idx)
                gpu_Gamma[right_idx, left_idx] = cp.ones(n_batch) / Z / Lambda[left_idx, i]
                # print(Gamma[right_idx, left_idx, j, i])
                continue
            # print(j, size, left_target, right_target, left_idx, right_idx)
            
            target = cp.zeros([n_batch, size], dtype='int32')
            target[:, :left_n_select] = cp.copy(left_target[:, :size][left_idx])
            right_target_chosen = cp.copy(right_target[:, -size:][right_idx])
            if size == 0:
                right_target_chosen = cp.zeros([n_batch, 0], dtype='int32')
            right_n_select = right_target_chosen.shape[1]
            non_zero_locations = cp.where(right_target_chosen != 0)
            right_target_chosen[non_zero_locations] += num.shape[1]
            # print(size, target[:, -right_n_select:].shape, right_target_chosen[right_idx].shape)
            # print('begin')
            # print(target, target[:, -right_n_select:], right_target_chosen)
            target[:, -right_n_select:] += right_target_chosen
            target = cp.append(cp.zeros([n_batch, j], dtype=int), target, axis=1)
            
            # print(j, size)
            # print(target)
            
            denominator = cp.sqrt(factorial(j)) * left_denominator[left_idx] * right_denominator[right_idx]
            # cp.cuda.runtime.deviceSynchronize()
            tot_target_time += time.time() - start
            print(n_batch, size)
            start = time.time()
            haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
            # cp.cuda.runtime.deviceSynchronize()
            tot_a_elem_time += time.time() - start
            start = time.time()
            tot_haf_time += haf_time
            tot_sigma_time += sigma_time
            
            # print(j, size)
            # print(haf)
            gpu_Gamma[right_idx, left_idx] = haf / Z / Lambda[left_idx, i]
            # cp.cuda.runtime.deviceSynchronize()
            tot_gamma_time += time.time() - start
            # print(Gamma[right_idx, left_idx, j, i])
        Gamma[:, :, j, i] = cp.asnumpy(gpu_Gamma)

    S_r = np.copy(S_l)
    res_pre = np.copy(res)
    num_pre = np.copy(num)
    right_target = cp.array(push_to_end(cp.asnumpy(left_target)))
    right_sum = left_sum
    right_denominator = left_denominator

    print('Total {}, target {}, a elem {}, haf {}, sigma {}, gamma {}, kron {}, prep {}'.format(time.time() - real_start, tot_target_time, tot_a_elem_time, tot_haf_time, tot_sigma_time, tot_gamma_time, tot_kron_time, tot_prep_time))

i = M - 1
S_l = np.zeros((0, 0))
U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
Z = np.sqrt(np.prod(np.cosh(sq)))
Sigma = get_Sigma(U2, sq, U1)
num_pre = num_pre[res_pre > err_tol].reshape(num_pre.shape[0], -1)
right_target = get_target(num_pre)
right_n_select = left_target.shape[1]
right_sum = cp.array(np.sum(num_pre, axis=1))
right_denominator = cp.sqrt(cp.product(cp.array(factorial(num_pre)), axis=1))
for j in np.arange(d):
    for size in np.arange(int(cp.nanmax(right_sum)) + 1):
        right_idx = cp.where(right_sum == size)[0]
        n_batch = right_idx.shape[0]
        if size == 0 and j == 0:
            # print('r, l ', right_idx, left_idx)
            Gamma[cp.asnumpy(right_idx), 0, j, i] = cp.asnumpy(cp.ones(n_batch) / Z)
            continue

        target = cp.copy(right_target[:, -size:][right_idx])
        if size == 0:
            target = cp.zeros([n_batch, 0], dtype='int32')
        target = cp.append(cp.zeros([n_batch, j], dtype=int), target, axis=1)
        denominator = cp.sqrt(factorial(j)) * right_denominator[right_idx]
        haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
        Gamma[cp.asnumpy(right_idx), 0, j, i] = cp.asnumpy(haf / Z)

print('Total hafnian time: ', tot_haf_time)
print('Largest Sigma2 dimension: ', largest_sigma2_dim)
np.save('Gamma_3a.npy', Gamma)
np.save('Lambda_3a.npy', Lambda)
gpu_Gamma = Gamma
gpu_Lambda = Lambda
quit()



def blochmessiah(S):
    N, _ = S.shape

    if not is_symplectic(S):
        raise ValueError("Input matrix is not symplectic.")

    # Changing Basis
    R = (1 / np.sqrt(2)) * np.block(
        [[np.eye(N // 2), 1j * np.eye(N // 2)], [np.eye(N // 2), -1j * np.eye(N // 2)]]
    )
    Sc = R @ S @ np.conjugate(R).T
    # Polar Decomposition
    u1, d1, v1 = np.linalg.svd(Sc)
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

def cartesian(array1, array2):
    # array1_orig = array1
    # array2_orig = array2
    array1 = np.array(array1)
    array2 = np.array(array2)
    if len(array1.shape) == 1:
        array1 = array1.reshape(-1, 1)
    if len(array2.shape) == 1:
        array2 = array2.reshape(-1, 1)
    len1 = array1.shape[0]
    len2 = array2.shape[0]
    array1 = np.repeat(array1, len2, 0)
    array2 = np.tile(array2, (len1, 1))
    # assert np.allclose(np.concatenate([array1, array2], axis=1), np.array([np.append(a, b) for a in array1_orig for b in array2_orig]))
    return np.concatenate([array1, array2], axis=1)

    # return np.array([np.append(a, b) for a in array1 for b in array2])


def get_cumsum_kron(sq_cov, L, chi = 100, max_dim = 10 ** 5, cutoff = 6, err_tol = 10 ** (-12)):
    M = len(sq_cov) // 2
    mode = np.arange(L, M)
    modes = np.append(mode, mode + M)
    sq_cov_A = sq_cov[np.ix_(modes, modes)]

    D, S = williamson(sq_cov_A)
    d = (np.diag(D) - 1) / 2

    d[d < 0] = 0

    res = thermal_photons(d[0], cutoff)
    num = np.arange(cutoff)
    for i in range(1, M - L):
        res = np.kron(res, thermal_photons(d[i], cutoff))
        num = cartesian(num, np.arange(cutoff))
        num = num[res > err_tol]
        res = res[res > err_tol]
        idx = np.argsort(res)[-min(len(res), max_dim):]       
        #if len(res) > max_dim:
        res = res[idx][::-1]
        num = num[idx][::-1]
            
    len_ = min(chi, len(res))
    idx = np.argsort(res)[-len_:]
    idx_sorted = idx[np.argsort(res[idx])]
    res = res[idx_sorted][::-1]
    num = num[idx_sorted][::-1]

    return res, num, S

def get_Sigma(U2, sq, U1):
    M = len(sq)
    Sigma = np.zeros((2 * M, 2 * M), dtype = complex)
    Sigma[:M, :M] = U2 @ np.diag(np.tanh(sq)) @ U2.T
    Sigma[:M, M:] = U2 @ np.diag(1 / np.cosh(sq)) @ U1
    Sigma[M:, :M] = U1.T @ np.diag(1 / np.cosh(sq)) @ U2.T
    Sigma[M:, M:] = -U1.T @ np.diag(np.tanh(sq)) @ U1
    return Sigma

def A_elem(Sigma, n_l, n_r): 
    n_ = np.append(n_l, n_r)
    if np.sum(n_) == 0:
        return 1, 0
    Sigma2 = np.repeat(np.repeat(Sigma, n_, axis = 0), n_, axis = 1)
    start = time.time()
    haf = default_hafnian(Sigma2, method='recursive')
    # haf = 0
    haf_time = time.time() - start
    # print('time: ', haf_time, Sigma2)
    # if Sigma2.shape[0] > 4:
    #     raise
    return haf / np.sqrt(np.product(factorial(n_))), haf_time

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



Gamma = np.zeros((chi, chi, d, M), dtype = complex)
Lambda = np.zeros((chi, M - 1), dtype = float)

max_dim = 10 ** 5; err_tol = 10 ** (-10)


i = 0
_, S_r = williamson(sq_cov)
res, num, S_l = get_cumsum_kron(sq_cov, i + 1, max_dim = max_dim, chi = chi, cutoff = d)
num = num[res > err_tol]
res = res[res > err_tol]
print(np.sum(res))
# obtain the singular values and eigen state of the right hand side.

U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
Sigma = get_Sigma(U2, sq, U1)

Z = np.sqrt(np.prod(np.cosh(sq)))

tot_haf_time = 0
largest_sigma2_dim = 0

Lambda[:len(res), 0] = np.sqrt(res)
for j in np.arange(d):
    for r in np.arange(min(chi, len(res))):
        n_l = np.append([j], num[r])
        if Lambda[r, i] > err_tol:
            haf, haf_time = A_elem(Sigma, n_l, [0] * M)
            Gamma[0, r, j, i] = haf / Z / Lambda[r, i]
            tot_haf_time += haf_time
S_r = np.copy(S_l)

res_pre = np.copy(res)
num_pre = np.copy(num)

for i in range(1, M - 1):
    res, num, S_l = get_cumsum_kron(sq_cov, i + 1, max_dim = max_dim, chi = chi, cutoff = d)
    num = num[res > err_tol]
    # print(i, np.sum(num, axis=1).max())
    res = res[res > err_tol]    
    print(np.sum(res))
    U2, sq, U1 = get_U2_sq_U1(S_l, S_r) # S_l: left in equation, S_r : right in equation
    Sigma = get_Sigma(U2, sq, U1)
    Z = np.sqrt(np.prod(np.cosh(sq)))

    Lambda[:len(res), i] = np.sqrt(res)
    for j in np.arange(d): # photon number basis
        for l in np.arange(min(chi, len(res_pre))): # right in the equation.
            for r in np.arange(min(chi, len(res))): # left in the equation (this is pre)
                n_l = np.append([j], num[r]) # left in the equation
                if Lambda[r, i] > err_tol:          
                    # sigma2_dim = np.sum(n_l) + np.sum(num_pre[l])
                    # if sigma2_dim > largest_sigma2_dim:
                    #     largest_sigma2_dim = sigma2_dim
                    #     print(largest_sigma2_dim)
                    start = time.time()
                    haf, haf_time = A_elem(Sigma, n_l, num_pre[l])
                    # print(n_l, num_pre[l])
                    # if sigma2_dim == 12:
                    #     print('size 12 time: ', time.time() - start)
                    Gamma[l, r, j, i] = haf / Z / Lambda[r, i]
                    # print(Gamma[l, r, j, i])
                    tot_haf_time += haf_time
    S_r = np.copy(S_l)
    res_pre = np.copy(res)
    num_pre = np.copy(num)

i = M - 1
S_l = np.zeros((0, 0))
U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
Z = np.sqrt(np.prod(np.cosh(sq)))
Sigma = get_Sigma(U2, sq, U1)
for j in np.arange(d):
    for l in np.arange(min(chi, len(res_pre))):
        for r in np.arange(1):
            n_l = [j]
            haf, haf_time = A_elem(Sigma, n_l, num_pre[l])
            Gamma[l, r, j, i] = haf / Z
            tot_haf_time += haf_time

print('Total hafnian time: ', tot_haf_time)
print('Largest Sigma2 dimension: ', largest_sigma2_dim)
np.set_printoptions(threshold=np.inf)
np.save('cpu.npy', Gamma)

print(np.allclose(Lambda, gpu_Lambda))
for i in range(M):
    print(np.allclose(Gamma[:,:,:,:i], gpu_Gamma[:,:,:,:i], rtol=0.01, atol=10**(-5)))

# np.set_printoptions(precision=2)
# print(np.real(Gamma[:,:,:,2])-np.real(gpu_Gamma[:,:,:,2]))
# print(np.real(Gamma[:,:,:,2]), np.real(gpu_Gamma[:,:,:,2]))
# print(np.real(gpu_Gamma[:,:,:,0]))

'''
import numpy as np
cpu = np.load('cpu.npy')
gpu = np.load('gpu.npy')
cpu[:,:,:,-1] - gpu[:,:,:,-1]
'''