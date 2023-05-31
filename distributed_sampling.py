import numpy as np
import cupy as cp
from tqdm import tqdm
import argparse
import torch
from scipy.linalg import expm
import warnings
import sys
import os
# os.environ["CUPY_TF32"] = "1"

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

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, help='Total number of samples.')
parser.add_argument('--n', type=int, help='Number of samples per batch.')
parser.add_argument('--iter', type=int, help='Number of iterations of sampling.')
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--dd', type=int, help='d for after random displacement. Maximum number of photons per mode that can be sampled - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--dir', type=str, help="Root directory.")
parser.add_argument('--ls', type=str, help="Local scratch directory.")
parser.add_argument('--gpn', type=int, help="Number of GPUs per node")
args = vars(parser.parse_args())

N = args['N']
n = args['n']
iterations = args['iter']
d = args['d']
dd = args ['dd']
chi = args['chi']
rootdir = args['dir']
path = rootdir + f'd_{d}_chi_{chi}/'
local_scratch = args['ls']
gpn = args['gpn']

if not os.path.isdir(path) and rank==0:
    os.mkdir(path)

cp.cuda.Device(rank % gpn).use()

def nothing_function(object):
    return object


# Sampling operations on the first optical mode
def sampling_beginning(displacements, Gamma, Lambda, i):
    # For explanatory comments, see sampling_middle
    res = []
    req = None
    Gamma = cp.sum(Gamma, axis=0) # chi x dd
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
        random_thresholds = cp.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x dd
        has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x dd
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
    # Samples at most n samples in parallel, until N samples are generated
    for begin_batch in tqdm(range(0, N, n)):

        end_batch = min(N, begin_batch + n)
        samples_in_parallel = end_batch - begin_batch
        iteration_displacements = displacements[begin_batch : end_batch]

        pre_tensor = np.zeros([samples_in_parallel, chi], dtype='complex64')
        comm.Recv([pre_tensor, MPI.C_FLOAT_COMPLEX], source=rank-1, tag=0) # Receiving from previous node the vector
        pre_tensor = cp.array(pre_tensor, dtype='complex64')
        probs = []
        temp_tensor = pre_tensor * Lambda_pre # samples_in_parallel x chi
        temp_tensor = (temp_tensor @ Gamma).reshape(samples_in_parallel, chi, dd)
        temp_tensor = cp.einsum('Bmj,Bkj->Bmk', temp_tensor, iteration_displacements) # Batch-parallel matrix multiplication
        pre_tensor = cp.copy(temp_tensor)
        temp_tensor = cp.abs(temp_tensor) ** 2

        for j in range(dd):
            if rank == M - 1:
                probs.append(temp_tensor[:, 0, j])
            else:
                probs.append(cp.dot(temp_tensor[:, :, j], Lambda ** 2)); # appending shape samples_in_parallel
        
        # This block is for batch parallel weighted random choice
        random_thresholds = cp.array(np.random.rand(samples_in_parallel, 1)) # samples_in_parallel
        probs = cp.array(probs).T # samples_in_parallel x dd
        probs = probs / cp.sum(probs, axis=1)[:, np.newaxis] # samples_in_parallel x dd
        cumulative_probs = cp.cumsum(probs, axis=1) # samples_in_parallel x dd
        random_thresholds = cp.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x dd
        has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x dd
        n_photons = cp.sum(has_more_photons, axis=1) # samples_in_parallel
        res.append(cp.asnumpy(n_photons)) # Appending sampling results

        np.save(path + f'samples_site_{rank}_{i}.npy', np.array(res).astype('int8').T)

        if rank == M - 1:
            continue
        
        # Selecting entries of pre_tensor depending on the sampled outcome.
        batch_to_n_ph = cp.zeros([samples_in_parallel, dd], dtype='complex64')
        for n_ph in range(dd):
            batch_to_n_ph[cp.where(n_photons == n_ph)[0], n_ph] = 1
        pre_tensor = cp.asnumpy(cp.einsum('BmP, BP -> Bm', pre_tensor, batch_to_n_ph)) / pre_tensor.max().item() # division by max is needed because otherwise the propagated vector will have decreasing magnitude as it go through the chain of modes

        if req != None:
            req.wait()
        # Sending pre_tensor vector to the next mode
        req = comm.Isend([pre_tensor, MPI.C_FLOAT_COMPLEX], rank+1, tag=0)
    
    if req != None:
        req.wait()


def destroy(inputs):
    data = np.sqrt(np.arange(1, inputs, dtype='complex64'))
    return np.diag(data, 1);

def batch_displaces(dim, alpha): # N is the dim
    samples_in_parallel = alpha.shape[0]
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
    
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    thermal_cov = cov - sq_cov;
    thermal_cov = thermal_cov + 1.000001 * np.eye(len(thermal_cov)) * np.abs(np.min(np.linalg.eigvalsh(thermal_cov)))
    sqrtW = np.linalg.cholesky(thermal_cov)

    M = sqrtW.shape[0] // 2
    assert M == comm.Get_size()
    Lambda = None
    # Last mode (rank) does not need to have load Lambda from the right
    if rank != M - 1:
        Lambda = np.load(local_scratch + f'/Lambda_{rank}.npy') # Loading right Lambda
        req = comm.Isend([Lambda, MPI.FLOAT], rank + 1, tag=0) # Sending loaded Lambda to the next mode as its left Lambda
        Lambda = cp.array(Lambda, dtype='float32')
        Lambda = Lambda / cp.sum(cp.abs(Lambda)**2)
    # First mode (rank) does not need to receive Lambda from left
    if rank != 0:
        Lambda_pre = np.zeros(chi, dtype='float32')
        comm.Recv([Lambda_pre, MPI.FLOAT], source=rank - 1, tag=0) # Receiving left Lambda
        Lambda_pre = cp.array(Lambda_pre, dtype='float32')
        Lambda_pre = Lambda_pre / cp.sum(cp.abs(Lambda_pre)**2)
        tqdm = nothing_function
    if rank != M - 1:
        req.wait() # Synchronize upon completion of send

    Gamma_small = np.load(local_scratch + f'Gamma_{rank}.npy') # Load constructed MPS Gamma tensor with local Hilbert space dimension d (small)
    Gamma = np.zeros([chi, chi, dd], dtype='complex64') # Initialize MPS Gamma tensor that will store displaced Gamma. Larger local Hilbert space dimension dd
    Gamma[:, :, :d] = Gamma_small
    Gamma = cp.array(Gamma, dtype='complex64')

    # Repeat sampling for 'iterations' times
    for i in range(iterations):

        # rank 0 generates alphas needed for random displacement matrices
        if rank == 0:
            print('Generating random displacements')
            random_array = np.random.normal(size=(2 * M, N))
            pure_mu = sqrtW @ random_array
            pure_mu = pure_mu.T
            pure_alphas = batch_mu_to_alpha(pure_mu, hbar=2).astype('complex64')
            for target_rank in range(1, M):
                comm.Send([np.ascontiguousarray(pure_alphas[:, target_rank]), MPI.C_FLOAT_COMPLEX], target_rank, tag=0)
            pure_alpha = pure_alphas[:, 0]
        # other ranks receive the generated alphas
        else:
            pure_alpha = np.zeros(N, dtype='complex64')
            comm.Recv([pure_alpha, MPI.C_FLOAT_COMPLEX], source=0, tag=0)
        # generate displacement matrices from alphas
        displacements = batch_displaces(dd, pure_alpha)
        cp.get_default_memory_pool().free_all_blocks()

        if rank == 0:
            sampling_beginning(displacements, Gamma, Lambda, i)
        else:
            sampling_middle(M, displacements, Gamma, Lambda, Lambda_pre, i)