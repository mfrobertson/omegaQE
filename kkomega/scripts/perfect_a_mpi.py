from mpi4py import MPI
import numpy as np
from fisher import Fisher
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
        f = open(f"_outlogs/perf_a_run_{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()


def _main(exp, Nbins, Nell, dL2, Ntheta, out_dir, _id):
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
    _output(f"exp: {exp}, Nbins: {Nbins}, Nell: {Nell}, dL2: {dL2}, Ntheta: {Ntheta}", my_rank, _id)
    nu = 353e9

    _output("Initialising Fisher object...", my_rank, _id)
    fish = Fisher(exp=exp)
    Ls_samp = fish.covariance.get_log_sample_Ls(Lmin=30, Lmax=4000, Nells=Nell, dL_small=1)

    _output("Setting up parallelisation of workload...", my_rank, _id)

    N_steps = int((Nbins+1) * Nbins/2)
    workloads = _get_workloads(N_steps, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    indices = np.empty(N_steps, dtype=tuple)
    idx = 0
    for iii in range(Nbins):
        for jjj in range(iii, Nbins):
            indices[idx] = (iii, jjj)
            idx += 1


    _output("Getting bins...", my_rank, _id)
    cosmo = fish.covariance.power.cosmo
    Chi_max = cosmo.get_chi_star()
    bins = [cosmo.Chi_to_z(Chi_max / Nbins * iii) for iii in range(1, Nbins + 1)]
    gal_bins = np.empty(2 * Nbins)
    gal_bins[0] = 0
    gal_bins[1:] = np.repeat(bins, 2)[:-1]
    gal_bins = tuple(gal_bins)

    _output("Starting Fisher calculation...", my_rank, _id)

    start_time = MPI.Wtime()

    F_tot = 0
    my_indices = indices[my_start:my_end]
    for iii, jjj in my_indices:
        index_a1, index_a2 = iii * 2, iii * 2 + 1
        index_b1, index_b2 = jjj * 2, jjj * 2 + 1
        gal_bins_tmp = (gal_bins[index_a1], gal_bins[index_a2], gal_bins[index_b1], gal_bins[index_b2])
        F_tot += fish.get_bispectrum_Fisher("abw", Ls=Ls_samp, Ntheta=Ntheta, f_sky=0.4, gal_bins=gal_bins_tmp, gal_distro="perfect")
    end_time = MPI.Wtime()

    _output("Broadcasting results...", my_rank, _id)

    if my_rank == 0:
        print("Fisher time: " + str(end_time - start_time))
        _output("Fisher time: " + str(end_time - start_time), my_rank, _id)
        results_arr = np.ones(world_size)
        results_arr[my_rank] = F_tot
        for rank in range(1, world_size):
            F_tot = np.empty(1)
            world_comm.Recv([F_tot, MPI.DOUBLE], source=rank, tag=77)
            results_arr[rank] = F_tot[0]
        out_dir += f"/{exp}/{Nell}_{dL2}_{Ntheta}/{Nbins}/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        np.savetxt(out_dir+"/F", np.array([np.sqrt(np.sum(F_tot))]))
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        _output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([np.array([F_tot]), MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError("Arguments should be exp Nbins Nell dL2 Ntheta out_dir _id")
    exp = str(args[0])
    Nbins = int(args[1])
    Nell = int(args[2])
    dL2 = int(args[3])
    Ntheta = int(args[4])
    out_dir = str(args[5])
    _id = str(args[6])
    _main(exp, Nbins, Nell, dL2, Ntheta, out_dir, _id)
