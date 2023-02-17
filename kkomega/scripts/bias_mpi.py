from mpi4py import MPI
import numpy as np
from bias import bias
from cache.tools import parse_boolean
import os
import sys
import datetime


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


def _get_log_sample_Ls(Lmin, Lmax, Nells=500, dL_small=1):
    floaty = Lmax / 1000
    samp1 = np.arange(Lmin, floaty * 10, dL_small)
    samp2 = np.logspace(1, 3, Nells - np.size(samp1)) * floaty
    return np.concatenate((samp1, samp2))


def _output(message, my_rank, _id):
    if my_rank == 0:
        f = open(f"_outlogs/_bias_run_{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()


def _main(exp, N_Ls, dir, bi_typ, gmv, fields, _id):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    _output("-------------------------------------", my_rank, _id)
    _output(f"exp: {exp}, N_Ls: {N_Ls}, bi_typ: {bi_typ}, gmv: {gmv}, fields: {fields}", my_rank, _id)
    _output("Setting up parallisation of workload.", my_rank, _id)

    Ls = _get_log_sample_Ls(30, 3000, N_Ls)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    _output("Initialisation finished.", my_rank, _id)

    verbose = True if my_rank == 0 else False

    start_time = MPI.Wtime()
    N_A1, N_C1 = bias(Ls[my_start: my_end], bi_typ, curl=True, exp=exp, qe_fields=fields, gmv=gmv, N_L1=100, N_L3=200, Ntheta12=100, Ntheta13=100, F_L_path="_results/F_L_results", qe_setup_path=f"cache/_Cls/{exp}/Cls_cmb_6000.npy", verbose=verbose)
    end_time = MPI.Wtime()

    _output("Bias calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Bias time: " + str(end_time - start_time))
        _output("Bias time: " + str(end_time - start_time), my_rank, _id)
        N_arr = np.ones(N_Ls)
        N_arr[my_start: my_end] = N_A1 + N_C1
        for rank in range(1, world_size):
            start, end = _get_start_end(rank, workloads)
            N = np.empty(end - start)
            world_comm.Recv([N, MPI.DOUBLE], source=rank, tag=77)
            N_arr[start: end] = N
        gmv_str = "gmv" if gmv else "single"
        dir += f"{exp}/{fields}_{gmv_str}/{bi_typ}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir+"/Ls", Ls)
        np.save(dir+"/N", N_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        _output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        N = N_A1 + N_C1
        world_comm.Send([N, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError("Must supply arguments: exp bi_typ fields gmv Nell dir id")
    exp = str(args[0])
    bi_typ = str(args[1])
    fields = str(args[2])
    gmv = parse_boolean(args[3])
    N_Ls = int(args[4])
    dir = args[5]
    _id = args[6]
    _main(exp, N_Ls, dir, bi_typ, gmv, fields, _id)
