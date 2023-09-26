import numpy as np
from DEMNUnii.fields import Fields
import sys
import os
from omegaqe.tools import mpi, none_or_str, parse_boolean
import DEMNUnii


def setup_dirs(sims_dir, exp, deflect_typs):
    obs_typs = obs_typs = ["kappa", "omega"]
    for deflect_typ in deflect_typs:
        for obs_typ in obs_typs:
            full_dir = f"{sims_dir}/{deflect_typ}/{exp}/{obs_typ}"
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)


def main(exp, qe_typ, nsims, deflect_typ, iter, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, qe_typ: {qe_typ}, nsims: {nsims}, deflect_typ: {deflect_typ}, iter: {iter}, nthreads: {nthreads}", 0, _id)

    fields = Fields(exp, use_lss_cache=True, use_cmb_cache=True, nthreads=nthreads)
    deflect_typs = ["pbdem_dem", "pbdem_zero", "npbdem_dem", "diff_diff", "dem_dem"] if deflect_typ is None else [deflect_typ]
    sims_dir = DEMNUnii.SIMS_DIR
    qe_typ_str = qe_typ + "_iter" if iter else qe_typ
    setup_dirs(sims_dir, exp, deflect_typs)
    for sim in range(nsims):
        mpi.output(f"Sim: {sim}", 0, _id)
        for deflect_typ in deflect_typs:
            fields.setup_rec(sim, deflect_typ, iter=iter)
            kappa_rec = fields.get_kappa_rec(qe_typ, fft=False, iter=iter)
            fields.dm.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ_str}_{sim}.fits", kappa_rec)
            mpi.output(f"   {deflect_typ} kappa done.", 0, _id)
            omega_rec = fields.get_omega_rec(qe_typ, fft=False, iter=iter)
            fields.dm.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ_str}_{sim}.fits", omega_rec)
            mpi.output(f"   {deflect_typ} omega done.", 0, _id)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError(
            "Must supply arguments: exp qe_typ nsims deflect_typ iter nthreads _id")
    exp = str(args[0])
    qe_typ = str(args[1])
    nsims = int(args[2])
    deflect_typ = none_or_str(args[3])
    iter = parse_boolean(args[4])
    nthreads  = int(args[5])
    _id = str(args[6])
    main(exp, qe_typ, nsims, deflect_typ, iter, nthreads, _id)
