from DEMNUnii.fields import Fields
import sys
import os
from omegaqe.tools import mpi


def setup_dirs(sims_dir, exp, deflect_typs):
    obs_typs = obs_typs = ["kappa", "omega"]
    for deflect_typ in deflect_typs:
        for obs_typ in obs_typs:
            full_dir = f"{sims_dir}/{deflect_typ}/{exp}/{obs_typ}"
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)


def main(exp, qe_typ, nsims, sims_dir, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, qe_typ: {qe_typ}, nsims: {nsims}, sims_dir: {sims_dir}, nthreads: {nthreads}", 0, _id)

    fields = Fields(exp, use_lss_cache=True, use_cmb_cache=True, nthreads=nthreads)
    deflect_typs = ["demnunii", "diff_alpha", "diff_omega"]
    setup_dirs(sims_dir, exp, deflect_typs)
    for sim in range(nsims):
        mpi.output(f"Sim: {sim}", 0, _id)
        for deflect_typ in deflect_typs:
            fields.setup_rec(sim, deflect_typ)
            kappa_rec = fields.get_kappa_rec(qe_typ, fft=False)
            fields.dm.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ}_{sim}.fits", kappa_rec)
            mpi.output(f"   {deflect_typ} kappa done.", 0, _id)
            omega_rec = fields.get_omega_rec(qe_typ, fft=False)
            fields.dm.sht.write_map(f"{sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_{sim}.fits", omega_rec)
            mpi.output(f"   {deflect_typ} omega done.", 0, _id)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 6:
        raise ValueError(
            "Must supply arguments: exp qe_typ nsims sims_dir nthraeds _id")
    exp = str(args[0])
    qe_typ = str(args[1])
    nsims = int(args[2])
    sims_dir = str(args[3])
    nthreads  = int(args[4])
    _id = str(args[5])
    main(exp, qe_typ, nsims, sims_dir, nthreads, _id)
