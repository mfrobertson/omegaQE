from omegaqe.tools import mpi, parse_boolean, none_or_str
import sys
import os
from scipy import stats
import numpy as np
import fullsky_sims
from fullsky_sims.demnunii import Demnunii

dm = Demnunii()


def _save_ps(ps, bins, err, exp, nbins, nsims, deflect_typ, ext):
    ps_dir = f"{fullsky_sims.CACHE_DIR}/_ps/{deflect_typ}/{exp}/{nbins}"
    if not os.path.exists(ps_dir):
        os.makedirs(ps_dir)
    ps_filename = f"ps"
    err_filename = f"err"
    bins_filename = f"bins"
    extension = f"_{nsims}"
    extension += ext
    ps_full_path = f"{ps_dir}/{ps_filename}{extension}.npy"
    mpi.output(f"Saving cross-spectrum to {ps_full_path}", 0, _id)
    np.save(f"{ps_full_path}", ps)
    np.save(f"{ps_dir}/{err_filename}{extension}.npy", err)
    np.save(f"{ps_dir}/{bins_filename}{extension}.npy", bins)


def _bin_ps(Cl, nBins, Lmax=3000):
    Ls = np.arange(np.size(Cl))[:Lmax+1]
    Cl = Cl[:Lmax+1]
    means, bin_edges, binnumber = stats.binned_statistic(Ls, Cl, 'mean', bins=nBins)
    binSeperation = bin_edges[1] - bin_edges[0]
    kBins = np.asarray([bin_edges[i] - binSeperation / 2 for i in range(1, len(bin_edges))])
    counts, *others = stats.binned_statistic(Ls, Cl, 'count', bins=nBins)
    stds, *others = stats.binned_statistic(Ls, Cl, 'std', bins=nBins)
    errors = stds / np.sqrt(counts)
    return means, kBins, errors


def _get_mean_ps(exp, deflect_typ, tem_ext, nsims, nbins, nthreads):
    mpi.output(f"  sim: 0", 0, _id)
    omega_tem = dm.sht.read_map(f"{fullsky_sims.CACHE_DIR}/_tems/{deflect_typ}/{exp}/omega_tem_{0}{tem_ext}.fits")
    omega_rec = dm.sht.read_map(f"{fullsky_sims.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_{0}.fits")
    Cl_ww = dm.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
    Cl_ww_binned, bins, errs = _bin_ps(Cl_ww, nbins)
    Cl_ww_all = np.zeros((nsims,np.size(Cl_ww_binned)))
    Cl_ww_all[0] = Cl_ww_binned
    for sim in range(1,nsims):
        mpi.output(f"  sim: {sim}", 0, _id)
        omega_tem = dm.sht.read_map(f"{fullsky_sims.CACHE_DIR}/_tems/{deflect_typ}/{exp}/omega_tem_{sim}{tem_ext}.fits")
        omega_rec = dm.sht.read_map(f"{fullsky_sims.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_{sim}.fits")
        Cl_ww = dm.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
        Cl_ww_binned, bins, errs = _bin_ps(Cl_ww, nbins)
        Cl_ww_all[sim] = Cl_ww_binned
    ps = np.mean(Cl_ww_all, axis=0)
    err = np.std(Cl_ww_all, axis=0)/(np.sqrt(nsims))
    return ps, bins, err


def main(exp, tracer_noise, kappa_rec, qe_typ, nbins, nsims, deflect_typ, gauss_lss, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, nbins: {nbins}, nsims: {nsims}, deflect_typ: {deflect_typ}, gauss_lss: {gauss_lss}, nthreads: {nthreads}", 0, _id)

    deflect_typs = ["pbdem_dem", "pbdem_zero", "npbdem_dem", "diff_diff", "dem_dem"] if deflect_typ is None else [deflect_typ]

    ext = "_wN" if tracer_noise else ""
    if kappa_rec:
        ext += f"_{qe_typ}"
    if gauss_lss:
        ext += "_gauss"
    for deflect_typ in deflect_typs:
        mpi.output(f"Type: {deflect_typ}", 0, _id)
        ps, bins, err = _get_mean_ps(exp, deflect_typ, ext, nsims, nbins, nthreads)
        _save_ps(ps, bins, err, exp, nbins, nsims, deflect_typ, ext)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 10:
        raise ValueError(
            "Must supply arguments: exp tracer_noise kappa_rec qe_typ nbins nsims deflect_typ gauss_lss nthreads _id")
    exp = str(args[0])
    tracer_noise = parse_boolean(args[1])
    kappa_rec = parse_boolean(args[2])
    qe_typ = str(args[3])
    nbins = int(args[4])
    nsims = int(args[5])
    deflect_typ = none_or_str(args[6])
    gauss_lss = parse_boolean(args[7])
    nthreads = int(args[8])
    _id = str(args[9])
    main(exp, tracer_noise, kappa_rec, qe_typ, nbins, nsims, deflect_typ, gauss_lss, nthreads, _id)
