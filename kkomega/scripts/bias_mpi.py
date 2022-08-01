from mpi4py import MPI
import numpy as np
from bias import Bias
import os
import sys


def _get_workloads(N, world_size):
    workloads = [N // world_size for iii in range(world_size)]
    for iii in range(N % world_size):
        workloads[iii] += 1
    return workloads


def _get_start_end(my_rank, workloads):
    my_start = 0
    for iii in range(my_rank):
        my_start += workloads[iii]
    my_end = my_start + workloads[my_rank]
    return my_start, my_end


def _get_Ls(N_Ls):
    samp1 = np.arange(30, 40, 5)
    samp2 = np.logspace(1, 3, N_Ls-2) * 4
    return np.concatenate((samp1, samp2))

def _main(N_Ls):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    Ls = _get_Ls(N_Ls)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    N0_file = "cache/N0/N0_my_SO_14_14_TQU.npy"
    bias = Bias(N0_file, M_path="cache/M", init_qe=False)

    fields = bias.qe.parse_fields()

    Cls = np.load("cache/_Cls/Cls_cmb_4000.npy")
    for iii, field in enumerate(fields):
        lenCl = Cls[iii, 0, :]
        gradCl = Cls[iii, 1, :]
        N = Cls[iii, 2, :]
        bias.qe.initialise_manual(field, lenCl, gradCl, N)
    bias.qe.initialise()

    start_time = MPI.Wtime()
    N_A1_curl_TT, N_C1_curl_TT = bias.bias("TT", Ls[my_start: my_end],N_L1=10, N_L3=10, Ntheta12=10, Ntheta13=10)
    end_time = MPI.Wtime()

    if my_rank == 0:
        print("Bias time: " + str(end_time - start_time))
        N_arr = np.ones(N_Ls)
        N_arr[my_start: my_end] = N_A1_curl_TT + N_C1_curl_TT
        for rank in range(1, world_size):
            start, end = _get_start_end(rank, workloads)
            N = np.empty(end-start)
            world_comm.Recv([N, MPI.DOUBLE], source=rank, tag=77)
            N_arr[start: end] = N
        if not os.path.isdir("../../bias/tmp_run"):
            os.mkdir("../../bias/tmp_run")
        np.save("../../bias/tmp_run/Ls", Ls)
        np.save("../../bias/tmp_run/N", N_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
    else:
        N = N_A1_curl_TT + N_C1_curl_TT
        world_comm.Send([N, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    N_Ls = int(args[0])
    _main(N_Ls)