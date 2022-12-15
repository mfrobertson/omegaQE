from mpi4py import MPI
import numpy as np
from fisher import Fisher
from cache.tools import parse_boolean
import os
import sys
import datetime


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
    _output("Setting up parallisation of workload.", my_rank, _id)

    _output("Initialising Fisher object.", my_rank, _id)

    fish = Fisher()
    fish.setup_noise(exp=exp, qe=fields, gmv=gmv, ps="gradient", L_cuts=(30,3000,30,5000), iter=False, data_dir="data")
    fish.setup_bispectra(Nell=200)

    Ls = fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=Lmax, Nells=N_Ls, dL_small=2)

    workloads = _get_workloads(N_Ls, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    _output("Initialisation finished.", my_rank, _id)

    nu = 353e9

    start_time = MPI.Wtime()
    Ls, F_L = fish.get_F_L(typ, Ls[my_start: my_end], Nell2=NL2, Ntheta=Ntheta, nu=nu, return_C_inv=False, gal_distro="LSST_gold")
    end_time = MPI.Wtime()

    _output("Bias calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("F_L time: " + str(end_time - start_time))
        _output("F_L time: " + str(end_time - start_time), my_rank, _id)
        F_L_arr = np.ones(Ls)
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
