import cupy as cp
import time
import os
os.environ["CUPY_TF32"] = "1"


a = cp.ones([10000, 10000], dtype='float32')
a @ a
cp.cuda.runtime.deviceSynchronize()
start = time.time()
a @ a
cp.cuda.runtime.deviceSynchronize()
print(time.time() - start)