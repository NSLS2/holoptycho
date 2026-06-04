from mpi4py import MPI
comm = MPI.COMM_WORLD.Dup()
import numpy as np
import time

ar = np.array([0],dtype=np.float32)

for i in range(1000):
    if comm.Get_rank() == 0:
        ar[0] = i
    if comm.Get_rank() == 1:
        time.sleep(0.3)
    comm.Bcast(ar)
    print(f"{comm.Get_rank()} {ar[0]}")
