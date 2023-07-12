# Large-Scale Lossy Gaussian Boson Sampling Simulator
## Introduction
### Summary
This repository is for large-scale simulation of Gaussian boson sampling simulations, capable of simulating the largest quantum supremacy experiments to-date. We provide a supercomputer GPU implementation and a personal computer CPU only implementation.

We tested our supercomputer GPU implementation on the Polaris supercomputer at Argonne Leadership Computing Facility. It uses exactly $M$ (number of optical modes) GPUs (ranks). For experiments with the largest number of output photons (Jiuzhang2-P65-5, Jiuzhang3-high, Borealis-M216-high, Borealis-M288), we obtain the MPS with bond dimension 10000 in 10 minutes, and sample 10 million samples in approximately 60 minutes, totalling slightly over an hour. Our simulation out-performs all bench marks including correlations upto the 6th order.

We tested our CPU implementation on an M1 Macbook Pro. For the most difficult experiment to match the two point correlation (Jiuzhang2-P65-2, see explanation in italic below), we obtain the MPS after approximate 3 hours. The bond dimension of 2000 is chosen to match the two-point correlation slope with the experiment, although the quality of higher-order correlation decreases.

*The bond dimension required to match the two-point correlation slopes of all other experiments, including Jiuzhang2-P65-1,3,4,5, Jiuzhang3, and Borealis, are all smaller. This may be due to two factors:*
* *Increasing the squeezing parameter increases the experimental noise, hence reducing the two-point correlation slopes.*
* *Increasing the squeezing parameter increases the displacement channel strength. In general, the trace distance between the approximate state and the ideal state is smaller after both go through a noise channel such as the displacement channel, hence reducing the bond dimension requirement.*

### Requirements
The majority of the program is written using python, requiring the following python packages:

* numpy 
* scipy
* tqdm
* requests (for downloading data from Xanadu in `Xanadu_download.py`)
* pandas (for loading the squeezing parameters from xlsx file of Jiuzhang3 in `make_cov.py`)
* cvxpy and cvxopt (for obtaining the quantum-classical covariance matrix decomposition in  `get_decomposition.py`.)
* strawberryfields (data analysis only)
* thewalrus (data analysis only)

For GPU implementations, we need:
* cupy
* torch (for GPU matrix exponential, only used in sampling)
* mpi4py
* filelock

We recommend Anaconda for python virtual environment management, and it is best to start a clean environment for this project.

For threshold detector Torontonian analysis of Jiuzhang samples, a Julia program `compute_torontonian.jl` is used (https://arxiv.org/abs/2207.10058). Julia can also be installed using Anaconda within the same virtual environment.

### CuPy Issues
If GPU implementations are used, please insure that cupy is installed properly. Normally, you shouldn't have to install cudatoolkit before hand, as the default method of installing cupy from conda 
```bash
conda install -c conda-forge cupy
```
should take care of installing the appropriate version of cudatoolkit. For more details, see https://docs.cupy.dev/en/stable/install.html.

A custom cuda kernel is used for generating the MPS, and cupy has to compile the kernel. If a compilation error occurs stating that permission is denied to access a directory, this is becuase cupy does not have permission to access the default temp file directory. We can simply change the default temp file directory by changing the environment variable `CUPY_CACHE_DIR` (recommended) or `TMPDIR` (system-wide) to a directory with sufficient perimssion.

### MPI Issues

When an error occurs, the rank that experiences the error will abort, while other MPI ranks may not. If other ranks are waiting for communication to complete from that rank, the program will not exit and will wait indefinitely. This causes significant waste of allocated compute time, and we choose to perform all-rank abort whenever a single rank experiences some error. However, the drawback of this approach is that error stack cannot be properly traced and debugging is hard. For production, we recommend using the all-rank abort error handling to avoid accidental failed simulations from wasting time. During debugging and testing, we recommend turning off this behavior. In all GPU implementations, python scripts have a line of code that changes the default error handling behavior of python into the all-rank abort behavior:
```python
sys.excepthook = mpiabort_excepthook
```
You can comment such lines away to turn off this behavior.

The current implementation does not scale to 1000 GPUs or ranks on the Polaris system at Argonne Leadership Computing Facility. This may be an issue with mpi4py, the implementation of MPICH on Polaris, or our program, and has not been tested on other systems. We welcome pull requests if a fix is identified within our program, or instructions for addressing this issue in the Readme file.

MPI communications with cupy arrays has unexpected behaviors (therefore, we use numpy buffers exclusively for inter-rank communications). Not tested on other systems and therefore source of issue is unknown.

### Strawberryfields adn Thewalrus Issues
Although our simulation code base does not require the strawberryfields or thewalrus libraries, benchmarking may need them. However, we experimence some conflicts with other libraries, particularly breaking the symbols with numpy for certain linear algebra libraries such as Intel MKL. If such issue occurs, we recommend creating a separate environment for data analysis with the two libraries installed.

### PyTorch Removal
We use PyTorch in our sampling algorithm for matrix exponentiation on GPUs. This is needed when creating the displacement operator from the ladder operators. The benefit of using GPUs is less significant when the bond dimension is large, or when the number of samples to generate is small. It can be removed and substituted with cpu matrix exponentiation from `scipy.linalg.expm`.

### Tensor Core Acceleration
Most operations are tensor operations, and can potentially benefit from using Tensor Cores on Nvidia GPUs that offer siginificant performance gains. We experimented with enabling tensor cores by setting the environment variable `CUPY_TF32=1`, but no significant changes were observed. One limitation is `cp.einsum` (`cupy.einsum`), which may not be utilizing tensor cores. Further investigations are needed to determine if improvements are possible.

## Download Data

### Variable preparation
We prepare the variables specifying the location of the files and which experiment to run here. In the example given, rootdir is the location where experiment configuration data is all saved.
```bash
# Back slashes are needed at the end
rootdir='/grand/projects/QuantumSupremacy/' 
experiment='ustc_larger_5/'
dir="$rootdir""$experiment"
# Specify which file saves all python outputs.
# Error outputs are not redirected to outfile.
# Can be specified with job scheduler (if applicable)
outfile='output.out' 
```

Later when we specify the output file, we use `> $outfile` at the end of a command to create the file and write python printed outputs to the file, or use `>> $outfile` to append.

### Downloading
For downloading Borealis (Xanadu) data, execute the following python script:
```bash
python -u Xanadu_download.py --dir $rootdir >> outfile
```
This will download data to corresponding figure folders under `rootdir` (data used to create different figures). To choose which data to download, edit the `files_list` variable in `Xanadu_download.py`. Afterwards, we would like to rename the folders appropriately by executing the following commands (or rename using GUI file managers):
```bash
cd $rootdir # Go to root directory
# Renaming names for clarity and consistency
[ -d "fig2" ] && mv fig2 Borealis_M16
[ -d "fig3a" ] && mv fig3a Borealis_M216_low
[ -d "fig3b" ] && mv fig3b Borealis_M72
[ -d "fig4" ] && mv fig4 Borealis_M216_high
[ -d "figS15" ] && mv figS15 Borealis_M288
cd - # Return to the last directory
```
For downloading Jiuzhang2.0 data, go to http://quantum.ustc.edu.cn/web/node/951. If you would like to download it using the terminal, execute the following:
```bash
wget -P $rootdir \
http://quantum.ustc.edu.cn/web/sites/default/files/2021-07/GBS144%20Data.part1_.rar
wget -P $rootdir \
http://quantum.ustc.edu.cn/web/sites/default/files/2021-07/GBS144%20Data.part2_.rar
wget -P $rootdir \
http://quantum.ustc.edu.cn/web/sites/default/files/2021-07/GBS144%20Data.part3_.rar
wget -P $rootdir \
http://quantum.ustc.edu.cn/web/sites/default/files/2021-07/GBS144%20Data.part4_.rar
```
You can similarly decompress the compressed files in terminal:
```bash
# Unrar your files. `x` tells it to extract all parts.
unrar x 'GBS144 Data.part1_.rar'
```
If you do not have `sudo` privilege to install the `unrar` command, you can use the a container following this tutorial: https://vsoch.github.io/lessons/unrar-python/ (their python tutorial does not work as it still depends on the `unrar` executable). For example, if your system provides `singularity` for running containers, you can execute the following:
```bash
# Loading singulaity. Most linux servers have `module` command to manage modules. 
# If singularity is not available, try docker.
module load singularity
# Getting the container that can run unrar
singularity-ce pull --name rar.simg shub://singularityhub/rar
# If you are using a cluster, start an interactive session job.
# Using the image on a login node may not work
# The example is shown for slurm, but you may have a different scheduler
# and different arguments may be required as well
srun --nodes=1 --ntasks-per-node=1 --time=01:00:00 --pty bash
# Binding directory to container so that it can find the files
singularity shell -B <directory/that/has/your/rar/files>
# Unrar your files. `x` tells it to extract all parts.
unrar x 'GBS144 Data.part1_.rar'
# Exit singularity container
exit
# Exit interactive job
exit
```
If nothing works, just download to your own computer and extract there.

You should then rename data folders to `Jiuzhang2_P<P>_<id>` correctly, and you may use a GUI file manager or similar commands for the Xanadu data:
```bash
mv waist=65um/0.15 Jiuzhang2_P65_1
mv waist=65um/0.3 Jiuzhang2_P65_2
mv waist=65um/0.6 Jiuzhang2_P65_3
mv waist=65um/1.0 Jiuzhang2_P65_4
mv waist=65um/1.65 Jiuzhang2_P65_5
mv waist=125um/0.5 Jiuzhang2_P125_1
mv waist=125um/1.412 Jiuzhang2_P125_2
cp waist=65um/matrix.mtx Jiuzhang2_P65_1/matrix.mtx
cp waist=65um/matrix.mtx Jiuzhang2_P65_2/matrix.mtx
cp waist=65um/matrix.mtx Jiuzhang2_P65_3/matrix.mtx
cp waist=65um/matrix.mtx Jiuzhang2_P65_4/matrix.mtx
cp waist=65um/matrix.mtx Jiuzhang2_P65_5/matrix.mtx
cp waist=125um/matrix.mtx Jiuzhang2_P125_1/matrix.mtx
cp waist=125um/matrix.mtx Jiuzhang2_P125_2/matrix.mtx
```

## Running the Program

### Generating needed decomposition results
The first step is to generate the covariance matrix saved as `cov.npy`. Run the python script:
```bash
python -u make_cov.py --dir $dir >> outfile
```
The next step is to decompose the covariance matrix of the experiment into a pure quantum part and a random displacement part. The quantum part results in the covariance matrix saved as `sq_cov.npy`.
```bash
python -u get_decomposition.py --dir $dir >> $outfile
```

### Preparing Experimental Configuration

```bash
# Local Hilbert space dimension for constructing the MPS.
# Usually small because the actual squeezed photon number is small
d=4
# Local Hilbert space dimension for sampling from MPS after displacement.
# Larger because more photons are present in a mode after displacement.
dd=10 
chi=100 # Bond dimension
# Number of samples to save in a single numpy file per site.
# This is to prevent losing data when an error occurs,
# and is useful for rescuing partial progress.
N=1000
# Number of samples to produce in parallel.
# If set too large, results in GPU out-of-memory error.
n=100
# Number of iterations to produce N samples.
# In total, produce iter * N samples.
iter=10
```

### Supercomputing GPU Implementation

The GPU supercomputer implementation uses exactly $M$ (number of optical modes) GPUs (ranks). The program will not work is a different number of ranks is launched. Multiple ranks could be assigned to a single GPU, but this is not tested and is likely to cause out-of-memory errors.

Ensure that local scratch disk space is enabled. Otherwise, cannot save partial progress before sampling.

If no local scratch disk space is available, you need to change the python files 'distributed_kron.py' and 'distributed_MPS.py' to save partial progress to permenant storage, and also save with different filenames for different ranks. You will also need to change 'distributed_MPS.py' and 'distributed_sampling.py' to read rank-specific partial results produced by the previous program. Otherwise, you can implement a combined program that saves everything on volatile memory and not disk, but some GPU and CPU memory freeing and garbage collection operations might be needed.

Tensors are not saved, and only used for sampling. Only samples are saved to permenant storage. The samples are saved to individual files for different modes.


```bash
# Input appropriate job submission specifications. Ignored here

# Directory for local scratch space. Different systems may have different names.
# Not all systems have local scratch space and the code may need to be modified.
ls='/local/scratch/'
gpn=4 # Number of GPUs per node. Polaris has 4.

now=$(date +"%T")
echo "Start time : $now"

# Preparation of getting the MPS
# mpiexec parameters may be different depending on your MPI implementation
# (MPICH, Open MPI, etc. See instructions of your cluster.)
mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE --depth=$NDEPTH --cpu-bind depth \
--env OMP_NUM_THREADS=$NTHREADS -env OMP_PLACES=threads \
python -u distributed_kron.py --d $d --chi $chi \
--gpn $gpn --dir $dir --ls $ls >> $outfile

# Getting the MPS
mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE --depth=$NDEPTH --cpu-bind depth \
--env OMP_NUM_THREADS=$NTHREADS -env OMP_PLACES=threads \
python -u distributed_MPS.py --d $d --chi $chi \
--gpn $pgn --dir $dir --ls $ls >> $outfile

now=$(date +"%T")
echo "Finishing MPS time : $now"


mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE --depth=$NDEPTH --cpu-bind depth \
--env OMP_NUM_THREADS=$NTHREADS -env OMP_PLACES=threads \
python -u distributed_sampling.py --N $N --n $n --iter $iter --d $d \
--dd $dd --chi $chi --gpn $gpn --dir $dir --ls $ls >> $outfile

now=$(date +"%T")
echo "Finishing sampling time : $now"
```

### CPU Only Implementation

```bash
python kron_cpu.py --d $d --chi $chi --dir $rootdir
python MPS_cpu.py --d $d --chi $chi --dir $rootdir
python sampling_cpu.py --N $N --n $n --iter $iter --d $d --dd $dd --chi $chi --dir $rootdir
```

### Data Analysis
The analysis code is located in the `analysis` folder:
```bash
cd analysis
```
XEB_photon_click (XEB_photon_number)
Compute XEBs of samples.

Correlation_photon_click (correlation_photon_number)
Compute two-point correlation functions of photon click (photon number)

Tot_probs_photon_number (Tot_probs_photon_click)
Get the approximation of the probabilities of obtaining N photons (N clicks) for a given covariance matrix, which is needed to XEB for normalization.

For Jiuzhang data analysis, we compute torontonians of samples from different photon click sectors. It saves “tors_{N}.npy” for the N sector. The code from https://github.com/polyquantique/torontonian-julia is needed. We will clone the repository minimally:
```bash
git clone --depth=1 --branch=master https://github.com/polyquantique/torontonian-julia.git
rm -rf ./torontonian-julia/.git
```
We then run the program:
```bash
julia compute_torontonian.jl $dir <file/name/for/npz/samples>
```
where `<file/name/for/npz/samples>` should only be the file name as does not include the directory (will be taken care of by argument `dir`).

To load Jiuzhang2 and Jiuzhang3 data, use `Jiuzhang2_load_samples.py` and `Jiuzhang3_load_samples.py` to convert the provided binary files into numpy files.
