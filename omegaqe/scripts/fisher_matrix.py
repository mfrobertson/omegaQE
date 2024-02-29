from mpi4py import MPI
import numpy as np
import omegaqe
from omegaqe.fisher import Fisher
from omegaqe.tools import parse_boolean, mpi, none_or_str, getFileSep
import os
import sys

def _main(exp, typs, params, dir, _id):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    mpi.output("-------------------------------------", my_rank, _id)
    mpi.output(
        f"exp: {exp} ",
        my_rank, _id)
    mpi.output("Setting up parallisation of workload.", my_rank, _id)

    # params = np.array(["ombh2", "omch2", "omk", "tau", "As", "ns", "omnuh2", "w", "wa"])
    N_params = np.size(params)
    N_param_combos = N_params**2

    workloads = mpi.get_workloads(N_param_combos, world_size)
    my_start, my_end = mpi.get_start_end(my_rank, workloads)

    mpi.output("Initialisation finished.", my_rank, _id)

    fish = Fisher(exp=exp, qe="TEB", gmv=True, ps="gradient", L_cuts=(30,3000,30,5000), iter=False, iter_ext=False, data_dir=f"{omegaqe.DATA_DIR}")


    start_time = MPI.Wtime()
    F = np.empty(my_end-my_start)
    for idx, num in enumerate(np.arange(my_start, my_end)):
        iii = num // N_params
        jjj = num % 8
        F[idx] = fish.get_optimal_bispectrum_Fisher(typs, f_sky=0.4, param=(params[iii], params[jjj]), dx=None)

    end_time = MPI.Wtime()

    mpi.output("Fisher matrix calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Fisher matrix time: " + str(end_time - start_time))
        mpi.output("Fisher matrix time: " + str(end_time - start_time), my_rank, _id)
        F_arr = np.ones(N_param_combos)
        F_arr[my_start: my_end] = F
        for rank in range(1, world_size):
            start, end = mpi.get_start_end(rank, workloads)
            F = np.empty(end - start)
            world_comm.Recv([F, MPI.DOUBLE], source=rank, tag=77)
            F_arr[start: end] = F
        dir += f"{exp}/{typs}/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir + "/F", F_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        mpi.output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([F, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        raise ValueError(
            "Must supply arguments: exp typs, params, dir id")
    exp = str(args[0])
    typs = str(args[1])
    params = np.array(args[2].split(','))
    dir = args[3]
    _id = args[4]
    _main(exp, typs, params, dir, _id)
