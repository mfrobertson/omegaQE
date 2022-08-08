from mpi4py import MPI
import numpy as np
from bias import Bias
from cache.tools import parse_boolean
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

def _main(N_Ls, dir, bi_typ, gmv, fields):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    Ls = _get_Ls(N_Ls)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    N0_path = "cache/_N0"
    bias = Bias(N0_path, M_path="cache/_M", init_qe=False)

    parsed_fields_all = bias.qe.parse_fields()
    Cls = np.load("cache/_Cls/Cls_cmb_4000.npy")
    for iii, field in enumerate(parsed_fields_all):
        lenCl = Cls[iii, 0, :]
        gradCl = Cls[iii, 1, :]
        N = Cls[iii, 2, :]
        bias.qe.initialise_manual(field, lenCl, gradCl, N)
    bias.qe.initialise()

    nu = 353e9

    if my_rank == 0:
        bias.build_F_L(bi_typ, fields, gmv, nu)
        Cl_kk = bias.cache.Cl_kk
        Cov_kk = bias.cache.Cov_kk
        Cl_gk = bias.cache.Cl_gk
        Cl_Ik = bias.cache.Cl_Ik
        sample_F_L_Ls = bias.cache.sample_F_L_Ls
        F_L = bias.cache.F_L
        C_inv = bias.cache.C_inv
    else:
        Cl_kk = np.empty(4000)
        Cov_kk = np.empty(4000)
        Cl_gk = np.empty(4000)
        Cl_Ik = np.empty(4000)
        sample_F_L_Ls = np.empty(300)
        F_L = np.empty(300)
        C_inv = np.empty((len(bi_typ),len(bi_typ),4000))

    world_comm.Bcast(Cl_kk, root=0)
    world_comm.Bcast(Cov_kk, root=0)
    world_comm.Bcast(Cl_gk, root=0)
    world_comm.Bcast(Cl_Ik, root=0)
    world_comm.Bcast(sample_F_L_Ls, root=0)
    world_comm.Bcast(F_L, root=0)
    world_comm.Bcast(C_inv, root=0)

    if my_rank != 0:
        bias.cache.Cl_kk = Cl_kk
        bias.cache.Cov_kk = Cov_kk
        bias.cache.Cl_gk = Cl_gk
        bias.cache.Cl_Ik = Cl_Ik
        bias.cache.sample_F_L_Ls = sample_F_L_Ls
        bias.cache.F_L = F_L
        bias.cache.C_inv = C_inv
        bias.cache.typs = bi_typ
        bias.cache.fields = fields
        bias.cache.gmv = gmv
        bias.cache.nu = nu

    start_time = MPI.Wtime()
    N_A1_curl_TT, N_C1_curl_TT = bias.bias("TT", Ls[my_start: my_end], N_L1=10, N_L3=10, Ntheta12=10, Ntheta13=10)
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
        if not os.path.isdir(dir+"/tmp_run"):
            os.mkdir(dir+"/tmp_run")
        np.save(dir+"/tmp_run/Ls", Ls)
        np.save(dir+"/tmp_run/N", N_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
    else:
        N = N_A1_curl_TT + N_C1_curl_TT
        world_comm.Send([N, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 5:
        raise ValueError("Must supply arguments: bi_typ fields gmv Nell dir")
    bi_typ = str(args[0])
    fields = str(args[1])
    gmv = parse_boolean(args[2])
    N_Ls = int(args[3])
    dir = args[4]
    _main(N_Ls, dir, bi_typ, gmv, fields)