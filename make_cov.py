import numpy as np
import pandas as pd
import argparse
import re
from scipy.linalg import block_diag
from scipy.io import mmread

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help="Experiment directory.")
args = vars(parser.parse_args())

dir = args['dir']

'''Taken from thewalrus'''
def Qmat(cov, hbar=2):
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q

'''Taken from thewalrus'''
def Covmat(Q, hbar=2):
    # number of modes
    n = len(Q) // 2
    I = np.identity(n)
    N = Q[0:n, 0:n] - I
    M = Q[n : 2 * n, 0:n]
    mm11a = 2 * (N.real + M.real) + np.identity(n)
    mm22a = 2 * (N.real - M.real) + np.identity(n)
    mm12a = 2 * (M.imag + N.imag)
    cov = np.block([[mm11a, mm12a], [mm12a.T, mm22a]])

    return (hbar / 2) * cov

def cov_from_T(r_array, T): # for Xanadu
    cov0 = np.diag([np.exp(2 * r) for r in r_array] + [np.exp(-2 * r) for r in r_array]);
    Q_in = Qmat(cov0)
    
    Tc = T.conj()
    Tt = T.T
    Th = Tt.conj()
    a = block_diag(T, Tc)
    b = block_diag(Th, Tt)
    A = (np.eye(len(T) * 2) - a @ b)
    B = a @ Q_in @ b
    Q_out = A + B
    
    return Covmat(Q_out)

#From here, for USTC
def get_sigma_out(sigma_in, T):
    Tc = T.conj()
    Tt = T.T
    Th = Tt.conj()
    a = block_diag(T, Tc)
    b = block_diag(Th, Tt)
    A = 1 / 2 * (np.eye(len(T) * 2) - a @ b)
    B = a @ sigma_in @ b
    return A + B

def get_sigma_in(r_array):
    S11 = np.diag(np.repeat(np.cosh(r_array), 2))
    S12 = np.kron(np.diag(np.sinh(r_array)), [[0, 1], [1, 0]])
    S21 = S12
    S22 = S11
    S = np.block([[S11, S12], [S21, S22]])
    
    return S @ S.T / 2


borealis_regex = re.compile('Borealis')
jiuzhang2_regex = re.compile('Jiuzhang2')
jiuzhang3_regex = re.compile('Jiuzhang3')
if borealis_regex.search(dir):
    experiment = 'Borealis'
elif jiuzhang2_regex.search(dir):
    experiment = 'Jiuzhang2'
elif jiuzhang3_regex.search(dir):
    experiment = 'Jiuzhang3'
else:
    print(f'The directory {dir} doesn\'t match any know experiments (Borealis, Jiuzhang2, Jiuzhang3). Please make sure the name of the directory is correct.')
    quit()


if experiment == 'Borealis':
    # Xanadu case
    T = np.load(dir + "T.npy")
    r_array = np.load(dir + "r.npy")
    cov = cov_from_T(r_array, T)


elif experiment == 'Jiuzhang2':
    # r_array is from their server
    r_array = np.array([[0.453102,0.493469,0.542692,0.456608,0.545166,0.491912,0.541289,0.508052,0.4392,0.530726,0.521693,0.530614,0.485262,0.462239,0.455414,0.484168,0.521314,0.400673,0.470621,0.541239,0.53367,0.490161,0.483768,0.4837,0.51938],
                    [0.6319,0.693711,0.761626,0.639916,0.762987,0.684644,0.751102,0.708161,0.617473,0.738377,0.731625,0.742339,0.676561,0.649688,0.634656,0.669457,0.722032,0.561756,0.660365,0.753813,0.740236,0.688105,0.672253,0.674055,0.727417],
                    [0.890573,0.990229,1.08404,0.90913,1.08141,0.961887,1.05056,0.997362,0.881866,1.03694,1.04029,1.05156,0.953167,0.92732,0.893381,0.930467,1.00672,0.798793,0.93999,1.06044,1.03464,0.980106,0.942031,0.948877,1.03219],
                    [1.1457,1.28989,1.40783,1.17909,1.39897,1.23374,1.34205,1.28243,1.14986,1.32974,1.34988,1.35967,1.2259,1.20817,1.14791,1.18031,1.28181,1.03719,1.22134,1.36221,1.32073,1.27411,1.20506,1.21941,1.33682],
                    [1.45843,1.66477,1.81063,1.5149,1.79196,1.56538,1.69573,1.63187,1.48645,1.68722,1.73477,1.74074,1.56023,1.56007,1.45915,1.4781,1.61318,1.33484,1.57225,1.73176,1.66728,1.64086,1.52419,1.55053,1.71451]])

    T = mmread(dir + "/spoofing/USTC/larger/matrix.mtx")
    sigma_in = get_sigma_in(r_array[0])
    sigma = get_sigma_out(sigma_in, T)
    cov = Covmat(sigma + np.eye(len(sigma)) / 2)
    cov = (cov + cov.T) / 2


elif experiment == 'Jiuzhang3':
    # Jiuzhang3.0 to get combined
    df = pd.read_excel(dir + "sq_parameter/high_power.xlsx", header = None)
    r_array = np.array(df)
    T = mmread(dir + "matrix/high power.mtx")
    # To combine two squeezing parameter sets
    single_r_array = np.arcsinh(np.sqrt(np.sinh(r_array[:, 0]) ** 2 + np.sinh(r_array[:, 1]) ** 2))
    sigma_in = get_sigma_in(single_r_array)
    sigma = get_sigma_out(sigma_in, T)
    cov = Covmat(sigma + np.eye(len(sigma)) / 2)
    cov = (cov + cov.T) / 2


np.save(dir + 'cov.npy', cov)