from omegaqe.tools import mpi
import sys
import os
import numpy as np
import DEMNUnii
from DEMNUnii.demnunii import Demnunii

dm = Demnunii()


def _save_norms(norm_k, norm_w, exp, qe_typ, offset, nbins_k, nbins_w):
    norm_dir_k = f"{DEMNUnii.CACHE_DIR}/_iter_norm/{exp}/{qe_typ}/{offset}_{nbins_k}"
    if not os.path.exists(norm_dir_k):
        os.makedirs(norm_dir_k)
    norm_dir_w = f"{DEMNUnii.CACHE_DIR}/_iter_norm/{exp}/{qe_typ}/{offset}_{nbins_w}"
    if not os.path.exists(norm_dir_w):
        os.makedirs(norm_dir_w)
    norm_path_k = norm_dir_k + "/iter_norm_k.npy"
    norm_path_w = norm_dir_w + "/iter_norm_w.npy"
    mpi.output(f"Saving normalisations to {norm_dir_k} and {norm_dir_w}", 0, _id)
    np.save(f"{norm_path_k}", norm_k)
    np.save(f"{norm_path_w}", norm_w)


def main(exp, qe_typ, offset, nbins_k, nbins_w, nthreads, _id):
    dm.sht.nthreads=nthreads
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, , qe_typ: {qe_typ}, nthreads: {nthreads}", 0, _id)
    deflect_typ = "diff2_diff2"
    name_ext = "diff2"
    nsims = 40
    kappa_true = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/kappa_{name_ext}.fits")
    omega_true = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/omega_{name_ext}.fits")
    kappa_map = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/kappa/{qe_typ}_iter_{0}.fits")
    omega_map = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_iter_{0}.fits")
    cl_k_cross = dm.sht.map2cl(kappa_true, kappa_map)
    cl_w_cross = dm.sht.map2cl(omega_true, omega_map)
    for sim in np.arange(1, nsims):
        kappa_map = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/kappa/{qe_typ}_iter_{sim}.fits")
        omega_map = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_iter_{sim}.fits")
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
    _save_norms(norm_k_smooth, norm_w_smooth, exp, qe_typ, offset, nbins_k, nbins_w)

    


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError(
            "Must supply arguments: exp qe_typ offset nbins_k nbins_w nthreads _id")
    exp = str(args[0])
    qe_typ = str(args[1])
    offset = int(args[2])
    nbins_k = int(args[3])
    nbins_w = int(args[4])
    nthreads = int(args[5])
    _id = str(args[6])
    main(exp, qe_typ, offset, nbins_k, nbins_w, nthreads, _id)
