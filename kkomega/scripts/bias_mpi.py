from mpi4py import MPI
import numpy as np
from bias import Bias
from cache.tools import parse_boolean
import os
import sys
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline


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
    samp2 = np.logspace(1, 3, N_Ls - 2) * 4
    return np.concatenate((samp1, samp2))


def _output(message, my_rank, _id):
    if my_rank == 0:
        f = open(f"_output/_bias_run_{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()


def _main(exp, N_Ls, dir, bi_typ, gmv, fields, _id):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    if my_rank == 0:
        try:
            os.remove("_bias_run.out")
        except:
            pass

    _output("Setting up parallisation of workload.", my_rank, _id)

    Ls = _get_Ls(N_Ls)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    _output("Initialising Bias object.", my_rank, _id)

    N0_path = "cache/_N0"
    bias = Bias(N0_path, M_path="cache/_M", init_qe=False, exp=exp)

    _output("Setting up cached Cls for bias initialisation.", my_rank, _id)

    parsed_fields_all = bias.qe.parse_fields(includeBB=True)
    Cls = np.load(f"cache/_Cls/{exp}/Cls_cmb_6000.npy")
    for iii, field in enumerate(parsed_fields_all):
        lenCl = Cls[iii, 0, :]
        gradCl = Cls[iii, 1, :]
        N = Cls[iii, 2, :]
        bias.qe.initialise_manual(field, lenCl, gradCl, N)
    bias.qe.initialise()

    _output("Initialisation finished.", my_rank, _id)

    nu = 353e9

    if bi_typ != "theory":
        _output("Building F_L on root thread...", my_rank, _id)
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
            Cl_kk = np.empty(4001, dtype='d')
            Cov_kk = np.empty(4001, dtype='d')
            Cl_gk = np.empty(4001, dtype='d')
            Cl_Ik = np.empty(4001, dtype='d')
            sample_F_L_Ls = np.empty(300, dtype='d')
            F_L = np.empty(300, dtype='d')
            C_inv = np.empty((len(bi_typ), len(bi_typ), 4001), dtype='d')

        _output("F_L build finished. Broadcasting...", my_rank, _id)

        world_comm.Bcast([Cl_kk, MPI.DOUBLE], root=0)
        _output("  Cl_kk done.", my_rank, _id)
        world_comm.Bcast([Cov_kk, MPI.DOUBLE], root=0)
        _output("  Cov_kk done.", my_rank, _id)
        world_comm.Bcast([Cl_gk, MPI.DOUBLE], root=0)
        _output("  Cl_gk done.", my_rank, _id)
        world_comm.Bcast([Cl_Ik, MPI.DOUBLE], root=0)
        _output("  Cl_Ik done.", my_rank, _id)
        world_comm.Bcast([sample_F_L_Ls, MPI.DOUBLE], root=0)
        _output("  sample_F_L_Ls done.", my_rank, _id)
        world_comm.Bcast([F_L, MPI.DOUBLE], root=0)
        _output("  F_L done.", my_rank, _id)
        world_comm.Bcast([C_inv, MPI.DOUBLE], root=0)
        _output("  C_inv done.", my_rank, _id)

        _output("Broadcasting finished. Setting up broadcasted F_Ls...", my_rank, _id)

        if my_rank != 0:
            bias.cache.Cl_kk = Cl_kk
            bias.cache.Cov_kk = Cov_kk
            bias.cache.Cl_gk = Cl_gk
            bias.cache.Cl_Ik = Cl_Ik
            bias.cache.sample_F_L_Ls = sample_F_L_Ls
            bias.cache.F_L = F_L
            bias.cache.F_L_spline = InterpolatedUnivariateSpline(sample_F_L_Ls, F_L)
            bias.cache.C_inv = C_inv
            bias.cache.typs = bi_typ
            bias.cache.fields = fields
            bias.cache.gmv = gmv
            bias.cache.nu = nu

    _output("Setup complete. Calculating bias...", my_rank, _id)

    start_time = MPI.Wtime()
    N_A1_curl_TT, N_C1_curl_TT = bias.bias(bi_typ, fields, Ls[my_start: my_end], gmv=gmv, curl=True, N_L1=100, N_L3=100,Ntheta12=100, Ntheta13=100)
    end_time = MPI.Wtime()

    _output("Bias calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Bias time: " + str(end_time - start_time))
        _output("Bias time: " + str(end_time - start_time), my_rank, _id)
        N_arr = np.ones(N_Ls)
        N_arr[my_start: my_end] = N_A1_curl_TT + N_C1_curl_TT
        for rank in range(1, world_size):
            start, end = _get_start_end(rank, workloads)
            N = np.empty(end - start)
            world_comm.Recv([N, MPI.DOUBLE], source=rank, tag=77)
            N_arr[start: end] = N
        gmv_str = "gmv" if gmv else "single"
        dir += f"/{fields}_{gmv_str}/{bi_typ}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir, Ls)
        np.save(dir, N_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        _output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        N = N_A1_curl_TT + N_C1_curl_TT
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
