from DEMNUnii.fields import Fields
import healpy as hp
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


def main(exp, qe_typ, nsims, sims_dir, _id):
    log_file = f"_lens_obs_{exp}_{qe_typ}"
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, qe_typ: {qe_typ}, nsims: {nsims}, sims_dir: {sims_dir}", 0, log_file)

    fields = Fields(exp, use_lss_cache=True, use_cmb_cache=True)
    deflect_typs = ["demnunii", "diff_alpha"]
    setup_dirs(sims_dir, exp, deflect_typs)
    for sim in range(nsims):
        mpi.output(f"Sim: {sim}", 0, log_file)
        for deflect_typ in deflect_typs:
            fields.setup_rec(sim, deflect_typ)
            kappa_rec = fields.get_kappa_rec(qe_typ, fft=False)
            hp.write_map(f"{sims_dir}/{deflect_typ}/{exp}/kappa/{qe_typ}_{sim}.fits", kappa_rec)
            mpi.output(f"   {deflect_typ} kappa done.", 0, log_file)
            omega_rec = fields.get_omega_rec(qe_typ, fft=False)
            hp.write_map(f"{sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_{sim}.fits", omega_rec)
            mpi.output(f"   {deflect_typ} omega done.", 0, log_file)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 5:
        raise ValueError(
            "Must supply arguments: exp qe_typ nsims sims_dir _id")
    exp = str(args[0])
    qe_typ = str(args[1])
    nsims = int(args[2])
    sims_dir = str(args[3])
    _id = str(args[4])
    main(exp, qe_typ, nsims, sims_dir, _id)
