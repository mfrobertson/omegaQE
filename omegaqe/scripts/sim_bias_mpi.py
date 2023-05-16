from mpi4py import MPI
import numpy as np
import omegaqe
from omegaqe.fields import Fields
from omegaqe.tools import parse_boolean
import os
import sys
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline
import socket



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


def _output(message, my_rank, _id, use_rank=False):
    if my_rank == 0:
        f = open(f"_outlogs/_sim_bias_run_{_id}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()
    elif use_rank:
        f = open(f"_outlogs/_sim_bias_run_{_id}_{my_rank}.out", "a")
        f.write("[" + str(datetime.datetime.now()) + "] " + message + "\n")
        f.close()

def _C_inv_splines(field_labels, C_inv, L_max, L_min_cut, L_max_cut):
    N_fields = np.size(list(field_labels))
    C_inv_splines = np.empty((N_fields, N_fields), dtype=InterpolatedUnivariateSpline)
    Ls = np.arange(L_max+1)
    for iii in range(N_fields):
        for jjj in range(N_fields):
            C_inv_ij = C_inv[iii, jjj]
            C_inv_ij[L_max_cut+1:] = 0
            C_inv_ij[:L_min_cut] = 0
            C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls, C_inv_ij)
    return C_inv_splines


def _F_L(fields_labels, exp, gmv, fields):
    gmv_str = "gmv" if gmv else "single"
    Ls = np.load(f"{omegaqe.RESULTS_DIR}/F_L_results/{fields_labels}/{exp}/{gmv_str}/{fields}/30_3000/1_2000/Ls.npy")
    F_L = np.load(f"{omegaqe.RESULTS_DIR}/F_L_results/{fields_labels}/{exp}/{gmv_str}/{fields}/30_3000/1_2000/F_L.npy")
    return Ls, F_L


def _qe_typ(fields, gmv):
    if fields == "TT" and not gmv:
        return "T"
    if fields == "EB" and gmv:
        return "EB"
    if fields == "TEB" and gmv:
        return "TEB"
    raise ValueError(f"fields: {fields} and gmv: {gmv} combination not yet supported.")


def _main(exp, typ, LDres, HDres, maps, gmv, Nsims, Lmin_cut, Lmax_cut, use_kappa_rec, include_noise, gauss_cmb, diffMaps, out_dir, _id):
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
    _output(f"Node: {socket.gethostname()}", my_rank, _id, use_rank=True)
    _output(f"exp:{exp}, tracers:{typ}, LDres: {LDres}, HDres: {HDres}, fields:{maps}, gmv:{gmv}, Nsims: {Nsims}, kappa_rec: {use_kappa_rec}, include_noise: {include_noise}, gauss_cmb: {gauss_cmb}, diffMaps: {diffMaps}", my_rank, _id)
    nu = 353e9

    _output("    Preparing grad Cls...", my_rank, _id)
    if my_rank == 0:
        from omegaqe.cosmology import Cosmology
        cosmo = Cosmology()
        Tcmb = 2.7255
        conv_fac = (Tcmb*1e6)**2
        grad_tt = cosmo.get_grad_lens_ps('TT', 6000) * conv_fac
        grad_ee = cosmo.get_grad_lens_ps('EE', 6000) * conv_fac
        grad_bb = cosmo.get_grad_lens_ps('BB', 6000) * conv_fac
        grad_te = cosmo.get_grad_lens_ps('TE', 6000) * conv_fac
    else:
        grad_tt = np.empty(6001, dtype='d')
        grad_ee = np.empty(6001, dtype='d')
        grad_bb = np.empty(6001, dtype='d')
        grad_te = np.empty(6001, dtype='d')

    _output("    Broadcasting and storing grad Cls...", my_rank, _id)
    world_comm.Bcast([grad_tt, MPI.DOUBLE], root=0)
    world_comm.Bcast([grad_ee, MPI.DOUBLE], root=0)
    world_comm.Bcast([grad_bb, MPI.DOUBLE], root=0)
    world_comm.Bcast([grad_te, MPI.DOUBLE], root=0)
    resp_cls = dict.fromkeys(['tt', 'ee', 'te', 'bb'])
    resp_cls['tt'] = grad_tt
    resp_cls['ee'] = grad_ee
    resp_cls['bb'] = grad_bb
    resp_cls['te'] = grad_te

    workloads = _get_workloads(Nsims, world_size)
    my_start, my_end = _get_start_end(my_rank, workloads)

    _output(f"Initialising Fields object...", my_rank, _id, use_rank=True)
    field_obj = Fields(typ, exp=exp, N_pix_pow=LDres, setup_cmb_lens_rec=True, HDres=HDres, Nsims=2*Nsims, sim=0, resp_cls=resp_cls)

    _output(f"Setting up noise...", my_rank, _id)
    field_obj.setup_noise(exp=exp, qe=maps, gmv=gmv, ps="gradient", L_cuts=(30, 3000, 30, 5000), iter=False, iter_ext=False, data_dir=omegaqe.DATA_DIR)

    Lmax_C_inv = 5000  # Highest L N0 calculated to
    N_typs = np.size(list(typ))
    C_inv = np.empty((N_typs, N_typs, Lmax_C_inv + 1), dtype='d')

    if my_rank == 0:
        _output("    Preparing C_inv...", my_rank, _id)
        C_inv = field_obj.fish.covariance.get_C_inv(typ, Lmax_C_inv, nu)

    _output("    Broadcasting and storing C_inv...", my_rank, _id)
    world_comm.Bcast([C_inv, MPI.DOUBLE], root=0)

    C_inv_splines = _C_inv_splines(typ, C_inv, Lmax_C_inv, Lmin_cut, Lmax_cut)

    Ls, F_L = _F_L(typ, exp, gmv, maps)
    F_L_spline = InterpolatedUnivariateSpline(Ls, F_L)

    qe_typ = _qe_typ(maps, gmv)

    ps_arr = None
    ps_arr_dp = None
    for iii, sim in enumerate(np.arange(my_start, my_end)):
        _output("Changing simulation: " + f" ({sim})", my_rank, _id, use_rank=True)
        field_obj.change_sim(int(sim))

        _output("Starting sim bias calculation..."+ f" ({sim})", my_rank, _id, use_rank=True)

        start_time = MPI.Wtime()
        omega_rec = field_obj.get_omega_rec(qe_typ, include_noise=include_noise, gaussCMB=gauss_cmb, diffSims=diffMaps, diffSim_offset=Nsims)
        end_time = MPI.Wtime()
        _output("Lensing reconstruction time: " + str(end_time - start_time) + f" ({sim})", my_rank, _id, use_rank=True)

        if not gauss_cmb and not diffMaps:
            start_time = MPI.Wtime()
            omega_rec_dp = field_obj.get_omega_rec(qe_typ, include_noise=include_noise, phi_idx=int(sim + Nsims))
            end_time = MPI.Wtime()
            _output("Lensing reconstruction time (different phi): " + str(end_time - start_time) + f" ({sim})", my_rank, _id, use_rank=True)

        start_time = MPI.Wtime()
        omega_temp = field_obj.get_omega_template(Nchi=100, F_L_spline=F_L_spline, C_inv_spline=C_inv_splines, reinitialise=True, use_kappa_rec=use_kappa_rec, tracer_noise=include_noise, gaussCMB=gauss_cmb, diffMaps=diffMaps, diffMaps_offset=Nsims)
        end_time = MPI.Wtime()
        _output("Template construction time: " + str(end_time - start_time)+ f" ({sim})", my_rank, _id, use_rank=True)

        _output(f"Calculating cross-spectrum ({sim})", my_rank, _id, use_rank=True)
        Ls, ps_tmp = field_obj.get_ps(omega_rec, omega_temp, kmin=Lmin_cut, kmax=Lmax_cut)

        if not gauss_cmb and not diffMaps:
            _output(f"Calculating cross-spectrum ({sim}) (different phi)", my_rank, _id, use_rank=True)
            Ls, ps_tmp_dp = field_obj.get_ps(omega_rec_dp, omega_temp, kmin=Lmin_cut, kmax=Lmax_cut)

        if ps_arr is None:
            ps_arr = np.zeros((my_end-my_start, np.size(ps_tmp)))
        if ps_arr_dp is None:
            ps_arr_dp = np.zeros((my_end - my_start, np.size(ps_tmp)))
        ps_arr[iii] = ps_tmp
        if not gauss_cmb and not diffMaps:
            ps_arr_dp[iii] = ps_tmp_dp

    _output("Broadcasting results...", my_rank, _id)

    if my_rank == 0:
        ps_all = np.zeros((Nsims, np.size(ps_arr[0])))
        ps_all[my_start:my_end] = ps_arr

        if not gauss_cmb and not diffMaps:
            ps_all_dp = np.zeros((Nsims, np.size(ps_arr_dp[0])))
            ps_all_dp[my_start:my_end] = ps_arr_dp
        for rank in range(1, world_size):
            start, end = _get_start_end(rank, workloads)
            ps_tmp = np.empty((end-start, np.size(ps_all[0])))
            world_comm.Recv([ps_tmp, MPI.DOUBLE], source=rank, tag=77)
            ps_all[start:end] = ps_tmp

            if not gauss_cmb and not diffMaps:
                ps_tmp_dp = np.empty((end - start, np.size(ps_all[0])))
                world_comm.Recv([ps_tmp_dp, MPI.DOUBLE], source=rank, tag=77)
                ps_all_dp[start:end] = ps_tmp_dp
        gmv_str = "gmv" if gmv else "single"
        kappa_rec_str = "kappa_rec" if use_kappa_rec else ""
        if include_noise:
            kappa_rec_str += "_wN"
        if gauss_cmb:
            kappa_rec_str += "_gaussCMB"
        if diffMaps:
            kappa_rec_str += "_diffMaps"
        out_dir += f"/{typ}/{exp}/{gmv_str}/{maps}/{LDres}_{HDres}/{Lmin_cut}_{Lmax_cut}/{Nsims}/{kappa_rec_str}/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir+"/Ls", Ls)
        np.save(out_dir+"/ps", ps_all)
        if not gauss_cmb and not diffMaps:
            np.save(out_dir + "/ps_dp", ps_all_dp)
        end_time_tot = MPI.Wtime()
        print("Total time: " + str(end_time_tot - start_time_tot))
        _output("Total time: " + str(end_time_tot - start_time_tot), my_rank, _id)
    else:
        world_comm.Send([ps_arr, MPI.DOUBLE], dest=0, tag=77)
        if not gauss_cmb and not diffMaps:
            world_comm.Send([ps_arr_dp, MPI.DOUBLE], dest=0, tag=77)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 15:
        raise ValueError("Arguments should be exp typ LDres HDres fields gmv Nsims Lmin_cut Lmax_cut kappa_rec include_noise gauss_cmb diffMaps out_dir _id")
    exp = str(args[0])
    typ = str(args[1])
    LDres = int(args[2])
    HDres = int(args[3])
    fields = str(args[4])
    gmv = parse_boolean(args[5])
    Nsims = int(args[6])
    Lmin_cut=int(args[7])
    Lmax_cut = int(args[8])
    use_kappa_rec = parse_boolean(args[9])
    include_noise = parse_boolean(args[10])
    gauss_cmb = parse_boolean(args[11])
    diffMaps = parse_boolean(args[12])
    out_dir = str(args[13])
    _id = str(args[14])
    _main(exp, typ, LDres, HDres, fields, gmv, Nsims, Lmin_cut, Lmax_cut, use_kappa_rec, include_noise, gauss_cmb, diffMaps, out_dir, _id)
