from mpi4py import MPI
import numpy as np
import omegaqe
from omegaqe.fisher import Fisher
from omegaqe.cosmology import Cosmology
from omegaqe.tools import parse_boolean, mpi, none_or_str, getFileSep
import os
import sys

def _main(exp, typs, params, condition, dir, _id, H0):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    mpi.output(f"World size: {world_size}", my_rank, _id)

    start_time_tot = MPI.Wtime()

    mpi.output("-------------------------------------", my_rank, _id)
    mpi.output(
        f"exp: {exp} ",
        my_rank, _id)
    mpi.output("Setting up parallisation of workload.", my_rank, _id)

    N_params = np.size(params)
    N_param_combos = N_params**2

    all_indices = np.empty(N_param_combos, dtype=object)
    count = 0
    for iii in range(N_params):
        for jjj in range(N_params):
            all_indices[count] = (iii,jjj)
            count += 1

    workloads = mpi.get_workloads(N_param_combos, world_size)
    my_start, my_end = mpi.get_start_end(my_rank, workloads)

    mpi.output("Initialisation finished.", my_rank, _id)

    cosmo = Cosmology(paramfile="Planck")
    fish = Fisher(exp=exp, qe="TEB", gmv=True, ps="gradient", L_cuts=(30,3000,30,5000), iter=False, iter_ext=False, data_dir=f"{omegaqe.dir_path}/data_planck/", cosmology=cosmo)
    
    param_str = ""
    for p in params:
        param_str += "_" + p
    condition_dir = f"{dir}/condition/{param_str}"
    if condition:
        dx_bi = 1/np.sqrt(np.load(f"{condition_dir}/bi.npy"))


    start_time = MPI.Wtime()
    F_bi = np.empty(my_end-my_start)
    for _i, idx in enumerate(np.arange(my_start, my_end)):
        iii, jjj = all_indices[idx]
        if condition:
            F_bi[_i] = fish.get_optimal_bispectrum_Fisher(typs, Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=(dx_bi[iii], dx_bi[jjj]), dx_absolute=True, H0=H0)
        else:
            F_bi[_i] = fish.get_optimal_bispectrum_Fisher(typs, Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=None, H0=H0)

    end_time = MPI.Wtime()

    mpi.output("Fisher matrix calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Fisher matrix time: " + str(end_time - start_time))
        mpi.output("Fisher matrix time: " + str(end_time - start_time), my_rank, _id)
        F_arr_bi = np.ones(N_param_combos)
        F_arr_bi[my_start: my_end] = F_bi
        for rank in range(1, world_size):
            start, end = mpi.get_start_end(rank, workloads)
            F_bi = np.empty(end - start)
            world_comm.Recv([F_bi, MPI.DOUBLE], source=rank, tag=77)
            F_arr_bi[start: end] = F_bi

        dir += f"{exp}/{typs}/{param_str}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir + "/F_bi_long", F_arr_bi)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        mpi.output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([F_bi, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError(
            "Must supply arguments: exp typs, params, condition, dir, id, H0")
    exp = str(args[0])
    typs = str(args[1])
    params = np.array(args[2].split(','))
    condition = parse_boolean(args[3])
    # tau_prior (see 2309.03021)
    dir = args[4]
    _id = args[5]
    H0 = parse_boolean(args[6])
    _main(exp, typs, params,condition, dir, _id, H0)
