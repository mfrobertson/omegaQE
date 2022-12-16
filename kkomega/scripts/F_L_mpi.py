from mpi4py import MPI
import numpy as np
from fisher import Fisher
from cache.tools import parse_boolean
import os
import sys
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline


def _get_workloads(N, world_size):
    workloads = [N // world_size for _ in range(world_size)]
    for iii in range(N % world_size):
        workloads[iii] += 1
    return workloads


def _get_start_end(my_rank, workloads):
    my_start = 0
    for iii in range(my_rank):
        my_start += workloads[iii]
    my_end = my_start + workloads[my_rank]
    return my_start, my_end


def _output(message, my_rank, _id):
    if my_rank == 0:
        f = open(f"_outlogs/_F_L_run_{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()


def _main(typ, exp, fields, gmv, Lmax, NL2, Ntheta, N_Ls, out_dir, _id):
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

    _output("-------------------------------------", my_rank, _id)
    _output(f"typ:{typ}, exp:{exp}, fields:{fields}, gmv:{gmv}, Lmax:{Lmax}, NL2:{NL2}, Ntheta:{Ntheta}, N_Ls:{N_Ls}", my_rank, _id)
    nu = 353e9

    _output("Initialising Fisher object...", my_rank, _id)
    fish = Fisher()

    _output("Setting up noise...", my_rank, _id)
    fish.setup_noise(exp=exp, qe=fields, gmv=gmv, ps="gradient", L_cuts=(30,3000,30,5000), iter=False, data_dir="data")

    _output("Setting up bispectra splines...", my_rank, _id)
    fish.setup_bispectra(Nell=200)

    _output("    Preparing C_inv...", my_rank, _id)
    if my_rank == 0:
        C_inv = fish.covariance.get_C_inv(typ, Lmax, nu)
    else:
        N_typs = np.size(list(typ))
        C_inv = np.empty((N_typs, N_typs, Lmax + 1), dtype='d')

    _output("    Broadcasting and storing C_inv...", my_rank, _id)
    world_comm.Bcast([C_inv, MPI.DOUBLE], root=0)
    fish.C_inv = C_inv

    _output("    Storing C_omega_spline...", my_rank, _id)
    C_omega = np.load("cache/_C_omega/C_omega.npy")
    omega_Ls = np.load("cache/_C_omega/Ls.npy")
    fish.C_omega_spline = InterpolatedUnivariateSpline(omega_Ls, C_omega)

    _output("Setting up parallelisation of workload...", my_rank, _id)

    Ls = fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=Lmax, Nells=N_Ls, dL_small=2)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    _output("Starting F_L calculation...", my_rank, _id)

    start_time = MPI.Wtime()
    Ls, F_L = fish.get_F_L(typ, Ls[my_start: my_end], Nell2=NL2, Ntheta=Ntheta, nu=nu, return_C_inv=False, gal_distro="LSST_gold", use_cache=True, Lmin=30, Lmax=5000)
    end_time = MPI.Wtime()

    _output("Broadcasting results...", my_rank, _id)

    if my_rank == 0:
        print("F_L time: " + str(end_time - start_time))
        _output("F_L time: " + str(end_time - start_time), my_rank, _id)
        F_L_arr = np.ones(N_Ls)
        F_L_arr[my_start: my_end] = F_L
        for rank in range(1, world_size):
            start, end = _get_start_end(rank, workloads)
            F_L = np.empty(end - start)
            world_comm.Recv([F_L, MPI.DOUBLE], source=rank, tag=77)
            F_L_arr[start: end] = F_L
        gmv_str = "gmv" if gmv else "single"
        out_dir += f"/{typ}/{exp}/{gmv_str}/{fields}/{Lmax}/{NL2}_{Ntheta}/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir+"/Ls", Ls)
        np.save(out_dir+"/F_L", F_L_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        _output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([F_L, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 10:
        raise ValueError("Arguments should be typ exp fields gmv Lmax NL2 Ntheta N_Ls out_dir _id")
    typ = str(args[0])
    exp = str(args[1])
    fields = str(args[2])
    gmv = parse_boolean(args[3])
    Lmax = int(args[4])
    NL2 = int(args[5])
    Ntheta = int(args[6])
    N_Ls = int(args[7])
    out_dir = args[8]
    _id = args[9]
    _main(typ, exp, fields, gmv, Lmax, NL2, Ntheta, N_Ls, out_dir, _id)