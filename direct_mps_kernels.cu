#include <cupy/complex.cuh>

extern "C" __global__ __launch_bounds__(256, 2)
void sigma_select(const int n_batch,
            const int n_select,
            const int n_len,
            const complex<double> *Sigma,
            const int *idx,
            complex<double> *Sigma2) {

    int n_idx = blockIdx.x * 4 + threadIdx.x;
    int m_idx = blockIdx.y * 4 + threadIdx.y;
    int l_idx = blockIdx.z * 16 + threadIdx.z;

    if (l_idx < n_batch && m_idx < n_select && n_idx < n_select) {
        int m_target = idx[l_idx * n_select + m_idx];
        int n_target = idx[l_idx * n_select + n_idx];
        Sigma2[l_idx * n_select * n_select + m_idx * n_select + n_idx] = Sigma[m_target * n_len + n_target];
    }
}

extern "C" __global__
void factorial(int *arr, int size) {

    int idx = blockIdx.x * 1024 + threadIdx.x;

    if (idx < size) {
        int value = arr[idx];
        if (value == 0 || value == 1) {
            arr[idx] = 1;
        }
        else {
            int result = 1;
            for (value = value; value > 0; --value) {
                result *= value;
            }
            arr[idx] = result;
        }
    }
}