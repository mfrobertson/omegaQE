from mpi4py import MPI
import numpy as np
import omegaqe
from omegaqe.fisher import Fisher
from omegaqe.cosmology import Cosmology
from omegaqe.tools import parse_boolean, mpi, none_or_str, getFileSep
import os
import sys

def _main(exp, typs, params, condition, dir, _id):
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

    # params = np.array(["ombh2", "omch2", "omk", "tau", "As", "ns", "omnuh2", "w", "wa"])
    N_params = np.size(params)
    # N_param_combos = N_params**2
    N_param_combos = int(N_params*(N_params-1)/2 + N_params)

    all_indices = np.empty(N_param_combos, dtype=object)
    count = 0
    for iii in range(N_params):
        for jjj in range(iii, N_params):
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
        dx_kk = 1/np.sqrt(np.load(f"{condition_dir}/kk.npy"))
        dx_cmb = 1/np.sqrt(np.load(f"{condition_dir}/cmb.npy"))


    start_time = MPI.Wtime()
    F_bi = np.empty(my_end-my_start)
    F_kk = np.empty(my_end - my_start)
    F_cmb = np.empty(my_end - my_start)
    F_kkk = np.empty(my_end - my_start)
    for _i, idx in enumerate(np.arange(my_start, my_end)):
        iii, jjj = all_indices[idx]
        if condition:
            F_bi[_i] = fish.get_optimal_bispectrum_Fisher(typs, Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=(dx_bi[iii], dx_bi[jjj]), dx_absolute=True)
            F_kk[_i] = fish.get_kappa_ps_Fisher(Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=(dx_kk[iii], dx_kk[jjj]), dx_absolute=True)
            F_cmb[_i] = fish.get_cmb_Fisher(Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=(dx_cmb[iii], dx_cmb[jjj]), dx_absolute=True)
        else:
            F_bi[_i] = fish.get_optimal_bispectrum_Fisher(typs, Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=None)
            F_kk[_i] = fish.get_kappa_ps_Fisher(Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=None)
            F_cmb[_i] = fish.get_cmb_Fisher(Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=None)
            F_kkk[_i] = fish.get_bispectrum_Fisher("kkk", Lmax=3000, f_sky=0.4, param=(params[iii], params[jjj]), dx=None)

    end_time = MPI.Wtime()

    mpi.output("Fisher matrix calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Fisher matrix time: " + str(end_time - start_time))
        mpi.output("Fisher matrix time: " + str(end_time - start_time), my_rank, _id)
        F_arr_bi = np.ones(N_param_combos)
        F_arr_kk = np.ones(N_param_combos)
        F_arr_cmb = np.ones(N_param_combos)
        F_arr_kkk = np.ones(N_param_combos)
        F_arr_bi[my_start: my_end] = F_bi
        F_arr_kk[my_start: my_end] = F_kk
        F_arr_cmb[my_start: my_end] = F_cmb
        F_arr_kkk[my_start: my_end] = F_kkk
        for rank in range(1, world_size):
            start, end = mpi.get_start_end(rank, workloads)
            F_bi = np.empty(end - start)
            world_comm.Recv([F_bi, MPI.DOUBLE], source=rank, tag=77)
            F_arr_bi[start: end] = F_bi

            F_kk = np.empty(end - start)
            world_comm.Recv([F_kk, MPI.DOUBLE], source=rank, tag=77)
            F_arr_kk[start: end] = F_kk

            F_cmb = np.empty(end - start)
            world_comm.Recv([F_cmb, MPI.DOUBLE], source=rank, tag=77)
            F_arr_cmb[start: end] = F_cmb

            F_kkk = np.empty(end - start)
            world_comm.Recv([F_kkk, MPI.DOUBLE], source=rank, tag=77)
            F_arr_kkk[start: end] = F_kkk
        dir += f"{exp}/{typs}/{param_str}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir + "/F_bi", F_arr_bi)
        np.save(dir + "/F_kk", F_arr_kk)
        np.save(dir + "/F_cmb", F_arr_cmb)
        np.save(dir + "/F_kkk", F_arr_kkk)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        mpi.output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([F_bi, MPI.DOUBLE], dest=0, tag=77)
        world_comm.Send([F_kk, MPI.DOUBLE], dest=0, tag=77)
        world_comm.Send([F_cmb, MPI.DOUBLE], dest=0, tag=77)
        world_comm.Send([F_kkk, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 6:
        raise ValueError(
            "Must supply arguments: exp typs, params, condition, dir id")
    exp = str(args[0])
    typs = str(args[1])
    params = np.array(args[2].split(','))
    condition = parse_boolean(args[3])
    # tau_prior (see 2309.03021)
    dir = args[4]
    _id = args[5]
    _main(exp, typs, params,condition, dir, _id)
