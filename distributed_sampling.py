import numpy as np
import cupy as cp
from tqdm import tqdm
import time
import argparse
import torch
from scipy.linalg import expm
import warnings
import sys
import os
os.environ["CUPY_TF32"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

def mpiabort_excepthook(type, value, traceback):
    print('type: ', type)
    print('value: ', value)
    print('traceback: ', traceback)
    print('An exception occured. Aborting MPI')
    comm.Abort()

sys.excepthook = mpiabort_excepthook

from decimal import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cp.cuda.Device(rank % 4).use()

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, help='Total number of samples.')
parser.add_argument('--n', type=int, help='Number of samples per batch.')
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--dd', type=int, help='d for after random displacement. Maximum number of photons per mode that can be sampled - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--rpn', type=int, help='Ranks per node. Should be the number of gpus available.')
parser.add_argument('--dir', type=str, help="Root directory.", default=0)
parser.add_argument('--iter', type=int, help='Number of iterations of sampling.')
args = vars(parser.parse_args())

N = args['N']
n = args['n']
d = args['d']
dd = args ['dd']
chi = args['chi']
rpn = args['rpn']
iterations = args['iter']
rootdir = args['dir']
path = rootdir + f'd_{d}_chi_{chi}/'
if not os.path.isdir(path) and rank==0:
    os.mkdir(path)

def nothing_function(object):
    return object



def sampling_beginning(displacements, Gamma, Lambda, i):

    res = []
    req = None
    # print('Gamma, displacements: ', Gamma.dtype, displacements.dtype)
    Gamma = cp.sum(Gamma, axis=0) # chi x cutoff
    for begin_batch in tqdm(range(0, N, n)):

        end_batch = min(N, begin_batch + n)
        samples_in_parallel = end_batch - begin_batch
        iteration_displacements = displacements[begin_batch : end_batch]
    
        random_thresholds = cp.array(np.random.rand(samples_in_parallel, 1)) # samples_in_parallel
        probs = []
        temp_tensor = cp.einsum('mj,Bkj->Bmk', Gamma, iteration_displacements)
        pre_tensor = cp.copy(temp_tensor)
        temp_tensor = cp.abs(temp_tensor) ** 2
        probs = [cp.dot(temp_tensor[:, :, j], Lambda ** 2) for j in range(dd)]
        probs = cp.array(probs).T
        probs = probs / cp.sum(probs, axis=1)[:, np.newaxis]
        cumulative_probs = cp.cumsum(probs, axis=1)
        random_thresholds = cp.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x cutoff
        has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x cutoff
        n_photons = cp.sum(has_more_photons, axis=1)
        res.append(cp.asnumpy(n_photons))
        batch_to_n_ph = cp.zeros([samples_in_parallel, dd], dtype='complex64')
        for n_ph in range(dd):
            batch_to_n_ph[cp.where(n_photons == n_ph)[0], n_ph] = 1
        pre_tensor = cp.einsum('BmP, BP -> Bm', pre_tensor, batch_to_n_ph)

        if req != None:
            req.wait()
        req = comm.Isend([pre_tensor, MPI.C_FLOAT_COMPLEX], rank+1, tag=0)

        np.save(path + f'samples_site_{rank}_{i}.npy', np.array(res).T)

    if req != None:
        req.wait()

def sampling_middle(M, displacements, Gamma, Lambda, Lambda_pre, i):

    res = []
    req = None
    Gamma = Gamma.reshape(chi, chi * dd)
    for begin_batch in tqdm(range(0, N, n)):

        end_batch = min(N, begin_batch + n)
        samples_in_parallel = end_batch - begin_batch
        iteration_displacements = displacements[begin_batch : end_batch]

        pre_tensor = np.zeros([samples_in_parallel, chi], dtype='complex64')
        comm.Recv([pre_tensor, MPI.C_FLOAT_COMPLEX], source=rank-1, tag=0)
        pre_tensor = cp.array(pre_tensor, dtype='complex64')
        probs = []
        temp_tensor = pre_tensor * Lambda_pre # samples_in_parallel x chi
        # temp_tensor = cp.einsum('Bn,nmj->Bmj', temp_tensor, Gamma) # samples_in_parallel x chi x cutoff
        temp_tensor = (temp_tensor @ Gamma).reshape(samples_in_parallel, chi, dd)
        temp_tensor = cp.einsum('Bmj,Bkj->Bmk', temp_tensor, iteration_displacements)
        pre_tensor = cp.copy(temp_tensor)
        temp_tensor = cp.abs(temp_tensor) ** 2

        for j in range(dd):
            if rank == M - 1:
                probs.append(temp_tensor[:, 0, j])
            else:
                probs.append(cp.dot(temp_tensor[:, :, j], Lambda ** 2)); # appending shape samples_in_parallel
        
        random_thresholds = cp.array(np.random.rand(samples_in_parallel, 1)) # samples_in_parallel
        probs = cp.array(probs).T # samples_in_parallel x cutoff
        probs = probs / cp.sum(probs, axis=1)[:, np.newaxis] # samples_in_parallel x cutoff
        cumulative_probs = cp.cumsum(probs, axis=1) # samples_in_parallel x cutoff
        random_thresholds = cp.array(random_thresholds)
        random_thresholds = cp.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x cutoff
        has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x cutoff
        n_photons = cp.sum(has_more_photons, axis=1) # samples_in_parallel
        res.append(cp.asnumpy(n_photons))

        np.save(path + f'samples_site_{rank}_{i}.npy', np.array(res).T)

        if rank == M - 1:
            continue

        batch_to_n_ph = cp.zeros([samples_in_parallel, dd], dtype='complex64')
        for n_ph in range(dd):
            batch_to_n_ph[cp.where(n_photons == n_ph)[0], n_ph] = 1
        pre_tensor = cp.asnumpy(cp.einsum('BmP, BP -> Bm', pre_tensor, batch_to_n_ph)) / pre_tensor.max().item()

        if req != None:
            req.wait()
        req = comm.Isend([pre_tensor, MPI.C_FLOAT_COMPLEX], rank+1, tag=0)
    
    if req != None:
        req.wait()


def destroy(inputs):
    data = np.sqrt(np.arange(1, inputs, dtype='complex64'))
    return np.diag(data, 1);

def displace(inputs, alpha): # N is the dim
    a = destroy(inputs)
    return expm(alpha * np.conj(a).T - np.conj(alpha) * a)

# def gpu_expm(mat):
#     mat = torch.tensor(mat, dtype=torch.complex64).cuda()
#     eigvals, eigvecs = torch.linalg.eig(mat)
#     eigvals = torch.exp(eigvals)
#     diag = eigvals.unsqueeze(1) * torch.eye(mat.shape[1], device='cuda')
#     return (eigvecs @ diag @ torch.permute(eigvecs.conj(), (0, 2, 1))).cpu().numpy()

def batch_displaces(dim, alpha): # N is the dim
    samples_in_parallel = alpha.shape[0]
    # alphas = alphas.reshape(-1, 1, 1)
    a = destroy(dim)
    a_h = np.conj(a).T
    a = np.repeat(a[np.newaxis], samples_in_parallel, axis=0)
    a_h = np.repeat(a_h[np.newaxis], samples_in_parallel, axis=0)
    alpha = alpha.reshape(-1, 1, 1)
    results = torch.linalg.matrix_exp(torch.tensor(alpha * a_h - np.conj(alpha) * a).cuda()).cpu().numpy()
    return results

def batch_mu_to_alpha(mu, hbar=2):
    M = mu.shape[1] // 2
    alpha = (mu[:, :M] + 1j * mu[:, M:]) / np.sqrt(2 * hbar)
    return alpha



if __name__ == "__main__":
    
    # sq_array = np.load(rootdir + "sq_array.npy")
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    thermal_cov = cov - sq_cov;
    thermal_cov = thermal_cov + 1.000001 * np.eye(len(thermal_cov)) * np.abs(np.min(np.linalg.eigvalsh(thermal_cov)))
    sqrtW = np.linalg.cholesky(thermal_cov)

    M = sqrtW.shape[0] // 2
    assert M == comm.Get_size()
    Lambda = None
    if rank != M - 1:
        Lambda = np.load(f'/local/scratch/Lambda_{rank}.npy')
        # Lambda = np.load(path + f'Lambda_{rank}.npy')
        req = comm.Isend([Lambda, MPI.FLOAT], rank + 1, tag=0)
        Lambda = cp.array(Lambda, dtype='float32')
        Lambda = Lambda / cp.sum(cp.abs(Lambda)**2)
    if rank != 0:
        Lambda_pre = np.zeros(chi, dtype='float32')
        comm.Recv([Lambda_pre, MPI.FLOAT], source=rank - 1, tag=0)
        Lambda_pre = cp.array(Lambda_pre, dtype='float32')
        Lambda_pre = Lambda_pre / cp.sum(cp.abs(Lambda_pre)**2)
        tqdm = nothing_function
    if rank != M - 1:
        req.wait()

    Gamma_small = np.load(f'/local/scratch/Gamma_{rank}.npy')
    # Gamma_small = np.load(path + f'Gamma_{rank}.npy')
    Gamma = np.zeros([chi, chi, dd], dtype='complex64')
    Gamma[:, :, :d] = Gamma_small
    Gamma = cp.array(Gamma, dtype='complex64')

    for i in range(iterations):

        if rank == 0:
            print('Generating random displacements')
            random_array = np.random.normal(size=(2 * M, N))
            pure_mu = sqrtW @ random_array
            pure_mu = pure_mu.T
            pure_alphas = batch_mu_to_alpha(pure_mu, hbar=2).astype('complex64')
            for target_rank in range(1, M):
                comm.Send([np.ascontiguousarray(pure_alphas[:, target_rank]), MPI.C_FLOAT_COMPLEX], target_rank, tag=0)
            pure_alpha = pure_alphas[:, 0]
        else:
            pure_alpha = np.zeros(N, dtype='complex64')
            comm.Recv([pure_alpha, MPI.C_FLOAT_COMPLEX], source=0, tag=0)
        displacements = batch_displaces(dd, pure_alpha)
        cp.get_default_memory_pool().free_all_blocks()

        # time.sleep(10)
        
        if rank == 0:
            sampling_beginning(displacements, Gamma, Lambda, i)
        else:
            sampling_middle(M, displacements, Gamma, Lambda, Lambda_pre, i)