from omegaqe.tools import mpi, parse_boolean
import sys
import os
import numpy as np
import fullsky_sims
from fullsky_sims.demnunii import Demnunii

dm = Demnunii()


def _save_norms(norm_k, norm_w, exp, qe_typ, offset, nbins_k, nbins_w, noise, gmv):
    norm_dir_k = f"{dm.cache_dir}/_iter_norm/{exp}/{qe_typ}_iter/{offset}_{nbins_k}"
    if not os.path.exists(norm_dir_k):
        os.makedirs(norm_dir_k)
    norm_dir_w = f"{dm.cache_dir}/_iter_norm/{exp}/{qe_typ}_iter/{offset}_{nbins_w}"
    if not os.path.exists(norm_dir_w):
        os.makedirs(norm_dir_w)
    ext = "_gmv" if gmv else ""
    norm_path_k = norm_dir_k + f"/iter_norm_k{ext}.npy"
    norm_path_w = norm_dir_w + f"/iter_norm_w{ext}.npy"
    mpi.output(f"Saving normalisations to {norm_dir_k} and {norm_dir_w}", 0, _id)
    np.save(f"{norm_path_k}", norm_k)
    np.save(f"{norm_path_w}", norm_w)


def main(exp, nsims, qe_typ, offset, nbins_k, nbins_w, noise, gmv, nthreads, _id):
    dm.sht.nthreads=nthreads
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, nsims: {nsims}, qe_typ: {qe_typ}, noise: {noise}, gmv: {gmv}, nthreads: {nthreads}", 0, _id)
    deflect_typ = "diff2_diff2"
    name_ext = "diff2"
    rec_ext = "" if noise else "nN"
    if gmv:
        rec_ext += "_gmv"
    kappa_true = dm.sht.read_map(f"{dm.sims_dir}/kappa_{name_ext}.fits")
    omega_true = dm.sht.read_map(f"{dm.sims_dir}/omega_{name_ext}.fits")
    kappa_map = dm.sht.read_map(f"{dm.sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ}_iter_{0}_{rec_ext}.fits")
    omega_map = dm.sht.read_map(f"{dm.sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_iter_{0}_{rec_ext}.fits")
    cl_k_cross = dm.sht.map2cl(kappa_true, kappa_map)
    cl_w_cross = dm.sht.map2cl(omega_true, omega_map)
    for sim in np.arange(1, nsims):
        kappa_map = dm.sht.read_map(f"{dm.sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ}_iter_{sim}_{rec_ext}.fits")
        omega_map = dm.sht.read_map(f"{dm.sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_iter_{sim}_{rec_ext}.fits")
        cl_k_cross += dm.sht.map2cl(kappa_true, kappa_map)
        cl_w_cross += dm.sht.map2cl(omega_true, omega_map)
        
    cl_k_true = dm.sht.map2cl(kappa_true)
    cl_w_true = dm.sht.map2cl(omega_true)
    norm_k = cl_k_cross/(nsims * cl_k_true)
    norm_w = cl_w_cross/(nsims * cl_w_true)
    norm_k_smooth = np.ones(np.size(norm_k))
    norm_w_smooth = np.ones(np.size(norm_w))
    norm_k_smooth[offset:] = dm.sht.smoothed_cl(norm_k[offset:], nbins=nbins_k, zerod=False)
    norm_w_smooth[offset:] = dm.sht.smoothed_cl(norm_w[offset:], nbins=nbins_w, zerod=False)
    _save_norms(norm_k_smooth, norm_w_smooth, exp, qe_typ, offset, nbins_k, nbins_w, noise, gmv)

    


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 10:
        raise ValueError(
            "Must supply arguments: exp nsims qe_typ offset nbins_k nbins_w noise gmv nthreads _id")
    exp = str(args[0])
    nsims = int(args[1])
    qe_typ = str(args[2])
    offset = int(args[3])
    nbins_k = int(args[4])
    nbins_w = int(args[5])
    noise = parse_boolean(args[6])
    gmv = parse_boolean(args[7])
    nthreads = int(args[8])
    _id = str(args[9])
    main(exp, nsims, qe_typ, offset, nbins_k, nbins_w, noise, gmv, nthreads, _id)
