from omegaqe.tools import mpi, parse_boolean, none_or_str
import sys
import os
import numpy as np
import DEMNUnii
from DEMNUnii.demnunii import Demnunii

dm = Demnunii()


def _save_ps(ps, exp, nsims, deflect_typ, ext):
    ps_dir = f"{DEMNUnii.CACHE_DIR}/_raw_ps/{deflect_typ}/{exp}/"
    if not os.path.exists(ps_dir):
        os.makedirs(ps_dir)
    ps_filename = f"ps"
    extension = f"_{nsims}"
    extension += ext
    ps_full_path = f"{ps_dir}/{ps_filename}{extension}.npy"
    mpi.output(f"Saving cross-spectrum to {ps_full_path}", 0, _id)
    np.save(f"{ps_full_path}", ps)


def _get_ps(exp, deflect_typ, tem_ext, nsims, nthreads):
    mpi.output(f"  sim: 0", 0, _id)
    omega_tem = dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}/_tems/{deflect_typ}/{exp}/omega_tem_{0}{tem_ext}.fits")
    omega_rec = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_{0}.fits")
    Cl_ww = dm.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
    for sim in range(1,nsims):
        mpi.output(f"  sim: {sim}", 0, _id)
        omega_tem = dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}/_tems/{deflect_typ}/{exp}/omega_tem_{sim}{tem_ext}.fits")
        omega_rec = dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/{deflect_typ}/{exp}/omega/{qe_typ}_{sim}.fits")
        Cl_ww += dm.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
    return Cl_ww/nsims


def main(exp, tracer_noise, kappa_rec, qe_typ, nsims, deflect_typ, gauss_lss, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, nsims: {nsims}, deflect_typ: {deflect_typ}, gauss_lss: {gauss_lss}, nthreads: {nthreads}", 0, _id)

    deflect_typs = ["pbdem_dem", "pbdem_zero", "npbdem_dem", "diff_diff", "dem_dem"] if deflect_typ is None else [deflect_typ]

    ext = "_wN" if tracer_noise else ""
    if kappa_rec:
        ext += f"_{qe_typ}"
    if gauss_lss:
        ext += "_gauss"
    for deflect_typ in deflect_typs:
        mpi.output(f"Type: {deflect_typ}", 0, _id)
        ps = _get_ps(exp, deflect_typ, ext, nsims, nthreads)
        _save_ps(ps, exp, nsims, deflect_typ, ext)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 9:
        raise ValueError(
            "Must supply arguments: exp tracer_noise kappa_rec qe_typ nsims deflect_typ gauss_lss nthreads _id")
    exp = str(args[0])
    tracer_noise = parse_boolean(args[1])
    kappa_rec = parse_boolean(args[2])
    qe_typ = str(args[3])
    nsims = int(args[4])
    deflect_typ = none_or_str(args[5])
    gauss_lss = parse_boolean(args[6])
    nthreads = int(args[7])
    _id = str(args[8])
    main(exp, tracer_noise, kappa_rec, qe_typ, nsims, deflect_typ, gauss_lss, nthreads, _id)