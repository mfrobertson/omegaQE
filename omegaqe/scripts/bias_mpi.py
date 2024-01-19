from mpi4py import MPI
import numpy as np
import omegaqe
from omegaqe.bias import bias
from omegaqe.tools import parse_boolean, mpi, none_or_str, getFileSep
import os
import sys


def _get_lss_cls_dict(cls_path):
    sep = getFileSep()
    cls_dict = {
        "kk": np.load(f"{cls_path}{sep}cl_kk.npy"),
        "gk": np.load(f"{cls_path}{sep}cl_gk.npy"),
        "Ik": np.load(f"{cls_path}{sep}cl_Ik.npy"),
    }
    return cls_dict

def _main(bias_typ, exp, N_Ls, N_L1, N_L3, Ntheta12, Ntheta13, noise, lss_cls_path, dir, bi_typ, gmv, fields, _id):
    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    start_time_tot = MPI.Wtime()

    mpi.output("-------------------------------------", my_rank, _id)
    mpi.output(f"bias_typ: {bias_typ}, exp: {exp}, N_Ls: {N_Ls}, N_L1: {N_L1}, N_L3: {N_L3}, Ntheta12: {Ntheta12}, Ntheta13: {Ntheta13}, noise: {noise}, lss_cls_path: {lss_cls_path}, bi_typ: {bi_typ}, gmv: {gmv}, fields: {fields}", my_rank, _id)
    mpi.output("Setting up parallisation of workload.", my_rank, _id)

    Ls = np.geomspace(30, 3000, N_Ls)

    workloads = mpi.get_workloads(N_Ls, world_size)
    my_start, my_end = mpi.get_start_end(my_rank, workloads)

    mpi.output("Initialisation finished.", my_rank, _id)

    verbose = True if my_rank == 0 else False

    
    lss_cls = None if lss_cls_path is None else _get_lss_cls_dict(lss_cls_path)
    
    if "_iter" in fields:
        iter = True
        qe_fields = fields[:-5]
        mpi.output(f"Iter rec on fields {qe_fields}.", my_rank, _id)
    else:
        iter=False
        qe_fields = fields
        mpi.output(f"QE rec on fields {qe_fields}.", my_rank, _id)

    start_time = MPI.Wtime()
    # TODO: be careful of F_L directory location
    N = bias(bias_typ, Ls[my_start: my_end], bi_typ, exp=exp, qe_fields=qe_fields, gmv=gmv, N_L1=N_L1, N_L3=N_L3, Ntheta12=Ntheta12, Ntheta13=Ntheta13, F_L_path=f"{omegaqe.CACHE_DIR}/_F_L", verbose=verbose, noise=noise, lss_cls=lss_cls, iter=iter)
    end_time = MPI.Wtime()

    mpi.output("Bias calculation finished.", my_rank, _id)

    if my_rank == 0:
        print("Bias time: " + str(end_time - start_time))
        mpi.output("Bias time: " + str(end_time - start_time), my_rank, _id)
        N_arr = np.ones(N_Ls)
        N_arr[my_start: my_end] = N
        for rank in range(1, world_size):
            start, end = mpi.get_start_end(rank, workloads)
            N = np.empty(end - start)
            world_comm.Recv([N, MPI.DOUBLE], source=rank, tag=77)
            N_arr[start: end] = N
        gmv_str = "gmv" if gmv else "single"
        bias_typ += "_nN" if not noise else ""
        dir += f"{exp}/{fields}_{gmv_str}/{bi_typ}/{bias_typ}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        np.save(dir+"/Ls", Ls)
        np.save(dir+"/N", N_arr)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        mpi.output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([N, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 14:
        raise ValueError("Must supply arguments: bias_typ exp bi_typ fields gmv Nell N_L1 N_L3 Ntheta12 Ntheta13 noise lss_cls_path dir id")
    bias_typ = str(args[0])
    exp = str(args[1])
    bi_typ = str(args[2])
    fields = str(args[3])
    gmv = parse_boolean(args[4])
    N_Ls = int(args[5])
    N_L1 = int(args[6])
    N_L3 = int(args[7])
    Ntheta12 = int(args[8])
    Ntheta13 = int(args[9])
    noise = parse_boolean(args[10])
    lss_cls_path = none_or_str(args[11])
    dir = args[12]
    _id = args[13]
    _main(bias_typ, exp, N_Ls, N_L1, N_L3, Ntheta12, Ntheta13, noise, lss_cls_path, dir, bi_typ, gmv, fields, _id)
