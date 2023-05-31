import numpy as np
import cupy as cp
from tqdm import tqdm
import argparse
import time
import sys
import os
os.environ["CUPY_TF32"] = "1"
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from scipy.special import factorial
from filelock import FileLock
from MPS_utils import williamson, get_U2_sq_U1, get_Sigma, get_target, A_elem, push_to_end

def nothing_function(object):
    return object

tqdm = nothing_function

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--gpn', type=int, help="GPUs per node.", default=0)
parser.add_argument('--dir', type=str, help="Root directory.", default=0)
parser.add_argument('--ls', type=str, help="Local scratch directory.")
args = vars(parser.parse_args())

d = args['d']
chi = args['chi']
gpn = args['gpn']
rootdir = args['dir']
path = rootdir + f'd_{d}_chi_{chi}/'
local_scratch = args['ls']
if not os.path.isdir(path) and rank==0:
    os.mkdir(path)

cp.cuda.Device(rank % gpn).use()



if __name__ == "__main__":

    def mpiabort_excepthook(type, value, traceback):
        print('type: ', type)
        print('value: ', value)
        print('traceback: ', traceback)
        print('An exception occured. Aborting MPI')
        comm.Abort()
    sys.excepthook = mpiabort_excepthook

    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    M = len(cov) // 2

    # idle_ranks.npy keeps track of which ranks are not busy with computation
    # This is to help determine where ranks in progress should send partial
    # computational load to
    if rank == 0:
        for site in range(M):
            if os.path.isfile(path + f'{site}.npy'):
                os.remove(path + f'{site}.npy')
        if not os.path.isdir(path):
            os.mkdir(path)
        idle_ranks = np.zeros(M, dtype='int32')
        np.save(path + 'idle_ranks.npy', idle_ranks)

    completed = True
    comm.bcast(completed, root=0)

    compute_site = rank
    real_start = time.time()

    max_memory_in_gb = 0.03
    max_dim = 10 ** 5; err_tol = 10 ** (-10)
    tot_a_elem_time = 0
    tot_haf_time = 0
    tot_sigma_time = 0

    _, S_r = williamson(sq_cov)

    Gamma = np.zeros([chi, chi, d], dtype='complex64')
    Lambda = cp.zeros([chi], dtype='float32')



    if compute_site == 0:

        res = np.load(local_scratch + f'res_{compute_site}.npy')
        num = np.load(local_scratch + f'num_{compute_site}.npy')
        S_l = np.load(local_scratch + f'S_{compute_site}.npy')
        num = num[res > err_tol]
        res = res[res > err_tol]
        U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
        Sigma = get_Sigma(U2, sq, U1)
        left_target = get_target(num)
        left_sum = np.sum(num, axis=1)
        left_denominator = cp.sqrt(cp.product(cp.array(factorial(num)), axis=1, dtype='float32'))
        Z = np.sqrt(np.prod(np.cosh(sq)))
        Lambda[:len(res)] = cp.array(np.sqrt(res))
        for j in np.arange(d):
            for size in np.arange(np.max(left_sum) + 1):
                left_idx = np.where(left_sum == size)[0]
                if (Lambda[left_idx] <= err_tol).all():
                    continue
                n_batch = left_idx.shape[0]
                '''one is already added to the left charge in function get_target'''
                target = cp.append(cp.zeros([n_batch, j], dtype='int32'), left_target[:, :size][left_idx], axis=1)
                denominator = cp.sqrt(factorial(j)) * left_denominator[left_idx]
                haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                tot_haf_time += haf_time
                Gamma[0, cp.asnumpy(left_idx), j] = cp.asnumpy(haf / Z / Lambda[left_idx])

    elif compute_site == M - 1:

        try:
            num_pre = np.load(local_scratch + f'num_{compute_site - 1}.npy')
        except:
            print('Failed. ', rank, compute_site - 1, os.listdir('/local/scratch/'))
            quit()
        num_pre = num_pre.reshape(num_pre.shape[0], -1)
        S_r = np.load(local_scratch + f'S_{compute_site - 1}.npy')
        right_target = get_target(num_pre)
        right_sum = cp.array(np.sum(num_pre, axis=1))
        right_denominator = cp.sqrt(cp.product(cp.array(factorial(num_pre)), axis=1))

        S_l = np.zeros((0, 0))
        U2, sq, U1 = get_U2_sq_U1(S_l, S_r)
        Z = np.sqrt(np.prod(np.cosh(sq)))
        Sigma = get_Sigma(U2, sq, U1)

        for j in np.arange(d):
            for size in np.arange(int(cp.nanmax(right_sum)) + 1):
                right_idx = cp.where(right_sum == size)[0]
                n_batch = right_idx.shape[0]
                if size == 0 and j == 0:
                    Gamma[cp.asnumpy(right_idx), 0, j] = cp.asnumpy(cp.ones(n_batch) / Z)
                    continue

                target = cp.copy(right_target[:, :size][right_idx])
                if size == 0:
                    target = cp.zeros([n_batch, 0], dtype='int32')
                target = cp.append(cp.zeros([n_batch, j], dtype=int), target, axis=1)
                denominator = cp.sqrt(factorial(j)) * right_denominator[right_idx]
                haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                Gamma[cp.asnumpy(right_idx), 0, j] = cp.asnumpy(haf / Z)

    else:
                
        num_pre = np.load(local_scratch + f'num_{compute_site - 1}.npy')
        res_pre = np.load(local_scratch + f'res_{compute_site - 1}.npy')
        S_r = np.load(local_scratch + f'S_{compute_site - 1}.npy')
        right_target = cp.array(push_to_end(cp.asnumpy(get_target(num_pre))))
        right_sum = cp.array(np.sum(num_pre, axis=1))
        right_denominator = cp.sqrt(cp.product(cp.array(factorial(num_pre)), axis=1, dtype='float32'))

        num = np.load(local_scratch + f'num_{compute_site}.npy')
        res = np.load(local_scratch + f'res_{compute_site}.npy')
        S_l = np.load(local_scratch + f'S_{compute_site}.npy')
        num = num[res > err_tol]
        num = num.reshape(num.shape[0], -1)
        left_target = get_target(num)
        left_n_select = left_target.shape[1]
        left_sum = cp.array(np.sum(num, axis=1))
        full_sum = cp.repeat(left_sum.reshape(-1, 1), right_sum.shape[0], axis=1) + cp.repeat(right_sum.reshape(1, -1), left_sum.shape[0], axis=0)
        left_denominator = cp.sqrt(cp.product(cp.array(factorial(num)), axis=1, dtype='float32'))
        res = res[res > err_tol]
        U2, sq, U1 = get_U2_sq_U1(S_l, S_r) # S_l: left in equation, S_r : right in equation
        Sigma = get_Sigma(U2, sq, U1)
        Z = np.sqrt(np.prod(np.cosh(sq)))
        Lambda[:len(res)] = cp.array(np.sqrt(res))

        for j in np.arange(d):
            gpu_Gamma = cp.zeros([chi, chi], dtype='complex64')
            for size in np.arange(int(cp.nanmax(full_sum)) + 1):
                left_idx, right_idx = cp.where(full_sum == size)
                n_batch = left_idx.shape[0]
                if (Lambda[left_idx] <= err_tol).all():
                    continue
                if size == 0 and j == 0:
                    gpu_Gamma[right_idx, left_idx] = cp.ones(n_batch) / Z / Lambda[left_idx]
                    continue
                if size == 0:
                    n_batch_max = 99999999999
                else:
                    n_batch_max = int(max_memory_in_gb * (10 ** 9) // (size * 8))
                requests = []
                buffers = []
                for begin_batch in tqdm(range(0, n_batch, n_batch_max)):
                    keep_going = True
                    while keep_going:
                        test_result = MPI.Request.Testany(requests)
                        completed_req = test_result[0]
                        if completed_req >= 0:
                            requests.pop(completed_req)
                            buffer = buffers.pop(completed_req)
                            haf, begin_batch, end_batch = buffer
                            gpu_Gamma[right_idx[begin_batch : end_batch], left_idx[begin_batch : end_batch]] = cp.array(haf) / Z / Lambda[left_idx[begin_batch : end_batch]]
                        else:
                            keep_going = False
                    end_batch = min(n_batch, begin_batch + n_batch_max)
                    target = cp.zeros([end_batch - begin_batch, size], dtype='int32')
                    target[:, :left_n_select] = cp.copy(left_target[:, :size][left_idx[begin_batch : end_batch]])
                    right_target_chosen = cp.copy(right_target[:, -size:][right_idx[begin_batch : end_batch]])
                    if size == 0:
                        right_target_chosen = cp.zeros([end_batch - begin_batch, 0], dtype='int32')
                    right_n_select = right_target_chosen.shape[1]
                    non_zero_locations = cp.where(right_target_chosen != 0)
                    right_target_chosen[non_zero_locations] += num.shape[1]
                    target[:, -right_n_select:] += right_target_chosen
                    target = cp.append(cp.zeros([end_batch - begin_batch, j], dtype='int32'), target, axis=1)
                    denominator = cp.array(cp.sqrt(factorial(j)) * left_denominator[left_idx[begin_batch : end_batch]] * right_denominator[right_idx[begin_batch : end_batch]], dtype='float32')
                    if end_batch != n_batch:
                        with FileLock(path + 'idle_ranks.npy.lock'):
                            idle_ranks = np.load(path + 'idle_ranks.npy')
                            available_ranks = np.where(idle_ranks == 1)[0]
                            target_rank = None
                            if available_ranks.shape[0] != 0:
                                idx = np.argmin(np.abs(available_ranks - rank))
                                target_rank = available_ranks[idx]
                                idle_ranks[target_rank] = 0
                                np.save(path + 'idle_ranks.npy', idle_ranks)
                        if target_rank != None:
                            try:
                                comm.send(True, target_rank, tag=100 + target_rank)
                            except:
                                print(rank, target_rank)
                                quit()
                            comm.send(rank, target_rank, tag=1)
                            comm.send(end_batch - begin_batch, target_rank, tag=2)
                            comm.send(size + j, target_rank, tag=3)
                            comm.send(Sigma.shape[0], target_rank, tag=4)
                            comm.Send([cp.asnumpy(Sigma), MPI.C_FLOAT_COMPLEX], target_rank, tag=5)
                            comm.Send([cp.asnumpy(target), MPI.INT], target_rank, tag=6)
                            comm.Send([cp.asnumpy(denominator), MPI.FLOAT], target_rank, tag=7)
                            haf = np.zeros([end_batch - begin_batch], dtype='complex64')
                            requests.append(comm.Irecv([haf, MPI.C_FLOAT_COMPLEX], target_rank, tag=8))
                            buffers.append([haf, begin_batch, end_batch])
                            continue
                    start = time.time()
                    haf, haf_time, sigma_time = A_elem(Sigma, target, denominator, max_memory_in_gb)
                    tot_a_elem_time += time.time() - start
                    tot_haf_time += haf_time
                    tot_sigma_time += sigma_time
                    gpu_Gamma[right_idx[begin_batch : end_batch], left_idx[begin_batch : end_batch]] = haf / Z / Lambda[left_idx[begin_batch : end_batch]]
                
                while len(requests) != 0:
                    time.sleep(0.01)
                    test_result = MPI.Request.Testany(requests)
                    completed_req = test_result[0]
                    if completed_req >= 0:
                        requests.pop(completed_req)
                        buffer = buffers.pop(completed_req)
                        haf, begin_batch, end_batch = buffer
                        gpu_Gamma[right_idx[begin_batch : end_batch], left_idx[begin_batch : end_batch]] = cp.array(haf) / Z / Lambda[left_idx[begin_batch : end_batch]]
                
            Gamma[:, :, j] = cp.asnumpy(gpu_Gamma)

    print('Total {}, a_elem {}, haf {}, sigma {}.'.format(time.time() - real_start, tot_a_elem_time, tot_haf_time, tot_sigma_time))

    np.save(local_scratch + f'Gamma_{compute_site}.npy', Gamma)
    np.save(local_scratch + f'Lambda_{compute_site}.npy', Lambda)
    print('Lambda: ', compute_site, cp.sum(cp.abs(Lambda)**2))
    # {compute_site}.npy indicates that computation for an optical mode has completed.
    np.save(path + f'{compute_site}.npy', np.ones(1))

    with FileLock(path + 'idle_ranks.npy.lock'):
        print(f'Rank {rank} done and acquired lock.')
        idle_ranks = np.load(path + 'idle_ranks.npy')
        idle_ranks[rank] = 1
        np.save(path + 'idle_ranks.npy', idle_ranks)

    def all_complete():
        for site in range(M):
            if not os.path.isfile(path + f'{site}.npy'):
                return False
        return True
    
    if all_complete():
        for target_rank in range(M):
            if target_rank == rank:
                continue
            comm.send(False, target_rank, tag=100+target_rank)
        print(f'rank {rank} completed MPS.')
        quit()
        
    keep_going = comm.recv(tag=100 + rank)
    while keep_going:
        source_rank = comm.recv(tag=1)
        n_batch = comm.recv(source=source_rank, tag=2)
        n_select = comm.recv(source=source_rank, tag=3)
        n_len = comm.recv(source=source_rank, tag=4)
        Sigma = np.zeros([n_len, n_len], dtype='complex64')
        target = np.zeros([n_batch, n_select], dtype='int32')
        denominator = np.zeros(n_batch, dtype='float32')
        comm.Recv([Sigma, MPI.C_FLOAT_COMPLEX], source=source_rank, tag=5)
        comm.Recv([target, MPI.INT], source=source_rank, tag=6)
        comm.Recv([denominator, MPI.FLOAT], source=source_rank, tag=7)
        haf, haf_time, sigma_time = A_elem(Sigma, target, cp.array(denominator), max_memory_in_gb)
        comm.Send([cp.asnumpy(haf), MPI.C_FLOAT_COMPLEX], dest=source_rank, tag=8)
        with FileLock(path + 'idle_ranks.npy.lock'):
            idle_ranks = np.load(path + 'idle_ranks.npy')
            idle_ranks[rank] = 1
            np.save(path + 'idle_ranks.npy', idle_ranks)
        try:
            keep_going = comm.recv(tag=100 + rank)
        except:
            print(rank)
            quit()
    print(f'rank {rank} completed MPS.')
