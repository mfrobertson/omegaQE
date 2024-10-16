import numpy as np
from fullsky_sims.fields import Fields
import sys
import os
from omegaqe.tools import mpi, none_or_str, parse_boolean
import fullsky_sims


def setup_dirs(sims_dir, exp, deflect_typs):
    obs_typs = ["kappa", "omega"]
    for deflect_typ in deflect_typs:
        for obs_typ in obs_typs:
            full_dir = f"{sims_dir}/{deflect_typ}/{exp}/{obs_typ}"
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)


def main(exp, qe_typ, start, end, deflect_typ, iter, noise, gmv, bh, nbody, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, qe_typ: {qe_typ}, start: {start}, end: {end}, deflect_typ: {deflect_typ}, iter: {iter}, noise: {noise}, gmv:{gmv}, bh:{bh},nbody: {nbody}, nthreads: {nthreads}", 0, _id)

    fields = Fields(exp, nbody, use_lss_cache=True, use_cmb_cache=True, nthreads=nthreads)
    deflect_typs = ["pbdem_dem", "pbdem_zero"] if deflect_typ is None else [deflect_typ]
    sims_dir = fields.nbody.sims_dir
    qe_typ_str = qe_typ + "_iter" if iter else qe_typ
    ext = "" if noise else "nN"
    if gmv:
        ext += "_gmv"
    if bh:
        ext += "_bh"
    setup_dirs(sims_dir, exp, deflect_typs)
    for sim in np.arange(start, end):
        mpi.output(f"Sim: {sim}", 0, _id)
        for deflect_typ in deflect_typs:
            fields.setup_rec(sim, deflect_typ, iter=iter, noise=noise, gmv=gmv)
            kappa_rec = fields.get_kappa_rec(qe_typ, fft=False, iter=iter, bias_hard=bh)
            fields.nbody.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ_str}_{sim}_{ext}.fits", kappa_rec)
            mpi.output(f"   {deflect_typ} kappa done.", 0, _id)
            omega_rec = fields.get_omega_rec(qe_typ, fft=False, iter=iter)
            fields.nbody.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ_str}_{sim}_{ext}.fits", omega_rec)
            mpi.output(f"   {deflect_typ} omega done.", 0, _id)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 12:
        raise ValueError(
            "Must supply arguments: exp qe_typ start end deflect_typ iter noise gmv bh nbody nthreads _id")
    exp = str(args[0])
    qe_typ = str(args[1])
    start = int(args[2])
    end = int(args[3])
    deflect_typ = none_or_str(args[4])
    iter = parse_boolean(args[5])
    noise = parse_boolean(args[6])
    gmv = parse_boolean(args[7])
    bh = parse_boolean(args[8])
    nbody = str(args[9])
    nthreads  = int(args[10])
    _id = str(args[11])
    main(exp, qe_typ, start, end, deflect_typ, iter, noise, gmv, bh, nbody, nthreads, _id)
