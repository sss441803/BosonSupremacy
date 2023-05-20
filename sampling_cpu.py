import numpy as np
from tqdm import tqdm
import argparse
from scipy.linalg import expm

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, help='d for calculating the MPS before random displacement. Maximum number of photons per mode before displacement - 1.')
parser.add_argument('--dd', type=int, help='d for after random displacement. Maximum number of photons per mode that can be sampled - 1.')
parser.add_argument('--chi', type=int, help='Bond dimension.')
parser.add_argument('--dir', type=str, help="Root directory.", default=0)
args = vars(parser.parse_args())

d = args['d']
dd = args ['dd']
chi = args['chi']
rootdir = args['dir']

def nothing_function(object):
    return object



def sampling(path, dd, Lambda, sqrtW, samples_in_parallel, compare=False):
    Gamma = np.load(path + f'Gamma_{0}.npy')
    print('ChiL: {}, d: {}.'.format(Gamma.shape[0], Gamma.shape[2]))
    d = Gamma.shape[2]
    M = len(sqrtW) // 2
    
    print('Generating random displacements')
    random_array = np.random.normal(size=(2 * M, samples_in_parallel))
    
    pure_mu = sqrtW @ random_array
    pure_mu = pure_mu.T
    pure_alpha = batch_mu_to_alpha(pure_mu, hbar=2)
    displacements = batch_displaces(dd, pure_alpha)

    np_res = []
    res = []
    #Sample
    for i in tqdm(range(M)):
        # if rank == 0:
        #     print(i)
        if i == 0:

            random_thresholds = np.random.rand(samples_in_parallel, 1) # samples_in_parallel

            probs = []
            temp_tensor = np.zeros([chi, chi, dd], dtype='complex64')
            temp_tensor[:, :, :d] = Gamma
            temp_tensor = np.sum(temp_tensor, axis=0) # chi x cutoff
            temp_tensor = np.einsum('mj,Bkj->Bmk', temp_tensor, displacements[:, i])
            pre_tensor = np.copy(temp_tensor)
            temp_tensor = np.abs(temp_tensor) ** 2
            probs = [np.dot(temp_tensor[:, :, j], Lambda[:, 0] ** 2) for j in range(dd)]
            probs = np.array(probs).T
            probs = probs / np.sum(probs, axis=1)[:, np.newaxis]
            cumulative_probs = np.cumsum(probs, axis=1)
            random_thresholds = np.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x cutoff
            has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x cutoff
            n_photons = np.sum(has_more_photons, axis=1)
            res.append(n_photons)
            batch_to_n_ph = np.zeros([samples_in_parallel, dd], dtype=int)
            for n_ph in range(dd):
                batch_to_n_ph[np.where(n_photons == n_ph)[0], n_ph] = 1
            pre_tensor = np.einsum('BmP, BP -> Bm', pre_tensor, batch_to_n_ph)
        else:
            probs = []
            tensor = pre_tensor * Lambda[:, len(res) - 1] # samples_in_parallel x chi
            Gamma = np.load(path + f'Gamma_{i}.npy')
            Gamma_temp = np.zeros([chi, chi, dd], dtype='complex64')
            Gamma_temp[:, :, :d] = Gamma
            temp_tensor = np.copy(tensor) # samples_in_parallel x chi
            temp_tensor = np.einsum('Bn,nmj->Bmj', temp_tensor, Gamma_temp) # samples_in_parallel x chi x cutoff
            temp_tensor = np.einsum('Bmj,Bkj->Bmk', temp_tensor, displacements[:, i])
            pre_tensor = np.copy(temp_tensor)
            temp_tensor = np.abs(temp_tensor) ** 2

            for j in range(dd):

                if len(res) == M - 1:
                    probs.append(temp_tensor[:, 0, j])
                else:
                    probs.append(np.dot(temp_tensor[:, :, j], Lambda[:, len(res)] ** 2)); # appending shape samples_in_parallel
            
            random_thresholds = np.array(np.random.rand(samples_in_parallel, 1)) # samples_in_parallel

            if compare:
                np_probs = np.array(np_probs) / np.sum(np_probs);     
                np_res.append(np.sum(np.cumsum(np_probs) < random_thresholds[0].item()))

            probs = np.array(probs).T # samples_in_parallel x cutoff
            probs = probs / np.sum(probs, axis=1)[:, np.newaxis] # samples_in_parallel x cutoff
            cumulative_probs = np.cumsum(probs, axis=1) # samples_in_parallel x cutoff
            random_thresholds = np.array(random_thresholds)
            random_thresholds = np.repeat(random_thresholds, dd, axis=1) # samples_in_parallel x cutoff
            has_more_photons = random_thresholds > cumulative_probs # samples_in_parallel x cutoff
            n_photons = np.sum(has_more_photons, axis=1) # samples_in_parallel
            res.append(n_photons)

            if i == M - 1:
                break
            
            batch_to_n_ph = np.zeros([samples_in_parallel, dd], dtype=int)
            for n_ph in range(dd):
                batch_to_n_ph[np.where(n_photons == n_ph)[0], n_ph] = 1
            pre_tensor = np.einsum('BmP, BP -> Bm', pre_tensor, batch_to_n_ph)

    results = np.array(res).T

    return results


def destroy(N):
    data = np.sqrt(np.arange(1, N, dtype=complex))
    return np.diag(data, 1);

def displace(N, alpha): # N is the dim
    a = destroy(N)
    return expm(alpha * np.conj(a).T - np.conj(alpha) * a)

def displaces(N, alphas): # N is the dim
    a = destroy(N)
    res = np.array([expm(alpha * np.conj(a).T - np.conj(alpha) * a) for alpha in alphas]);
    return np.array(res)

def batch_displaces(N, alphas): # N is the dim
    samples_in_parallel = alphas.shape[0]
    # alphas = alphas.reshape(-1, 1, 1)
    M = alphas.shape[1]
    a = destroy(N)
    a_h = np.conj(a).T
    a = np.repeat(a[np.newaxis], samples_in_parallel, axis=0)
    a_h = np.repeat(a_h[np.newaxis], samples_in_parallel, axis=0)
    results = []
    for i in tqdm(range(M)):
        alpha = alphas[:, i].reshape(-1, 1, 1)
        # results.append(torch.linalg.matrix_exp(torch.tensor(alpha * a_h - np.conj(alpha) * a).cuda()).cpu().numpy())
        results.append(expm(alpha * a_h - np.conj(alpha) * a))
    results = np.transpose(np.array(results), (1, 0, 2, 3))
    print(results.shape)
    return results

def mu_to_alpha(mu, hbar=2):
    M = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:M] + 1j * mu[M:]) / np.sqrt(2 * hbar)
    return alpha

def batch_mu_to_alpha(mu, hbar=2):
    M = mu.shape[1] // 2
    alpha = (mu[:, :M] + 1j * mu[:, M:]) / np.sqrt(2 * hbar)
    return alpha



if __name__ == "__main__":
    
    path = rootdir + f'd_{d}_chi_{chi}/'
    sq_array = np.load(rootdir + "sq_array.npy")
    sq_cov = np.load(rootdir + "sq_cov.npy")
    cov = np.load(rootdir + "cov.npy")
    thermal_cov = cov - sq_cov;
    thermal_cov = thermal_cov + 1.000001 * np.eye(len(thermal_cov)) * np.abs(np.min(np.linalg.eigvalsh(thermal_cov)))
    sqrtW = np.linalg.cholesky(thermal_cov)
    M = sqrtW.shape[0] // 2
    Lambda = np.zeros([chi, M - 1], dtype='float32')
    for i in range(M - 1):
        Lambda[:, i] = np.load(path + f"Lambda_{i}.npy")
    samples_per_rank = 5000
    
    samples = np.zeros([0, M], dtype='int8')
    # samples = np.load(rootdir + f"samples_d_{d}_dd_{dd}_chi_{chi}_{rank}.npy")
    for subsamples in range(1):
        subsamples = sampling(path, dd, Lambda, sqrtW, samples_per_rank, False)
        samples = np.concatenate([samples, subsamples], axis=0)
        np.save(rootdir + f"samples_d_{d}_dd_{dd}_chi_{chi}.npy", samples)
        print(samples.shape, samples.mean(), samples[:, 0].mean(), samples[:, -1].mean())
    # np.save(rootdir + f"samples_d_{d}_chi_{chi}_{rank}.npy", samples)