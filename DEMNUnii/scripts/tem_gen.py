from DEMNUnii.fields import Fields
from omegaqe.tools import mpi, parse_boolean
import sys
import healpy as hp
import os


def save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ):
    if not os.path.exists(tem_dir):
        os.makedirs(tem_dir)
    filename = f"omega_tem_{sim}"
    extension = "_wN" if tracer_noise else ""
    if kappa_rec:
        extension += f"_{kappa_qe_typ}"
    full_path = f"{tem_dir}/{filename}{extension}.fits"
    mpi.output(f"Saving template to {full_path}", 0, _id)
    hp.write_map(f"{full_path}", tem_map)


def main(exp, Nchi, tracer_noise, nsims, kappa_rec, kappa_qe_typ, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, Nchi: {Nchi}, nsim: {nsims}, tracer_noise: {tracer_noise}, kappa_rec: {kappa_rec}, kappa_qe_typ: {kappa_qe_typ}", 0, _id)

    deflect_typs = ["demnunii", "diff_alpha"]
    for sim in range(nsims):
        for deflect_typ in deflect_typs:
            fields = Fields(exp, use_lss_cache=True, use_cmb_cache=True, cmb_sim=sim, deflect_typ=deflect_typ)
            omega_tem = fields.omega_template(Nchi, tracer_noise=tracer_noise, use_kappa_rec=kappa_rec, kappa_rec_qe_typ=kappa_qe_typ)
            tem_map = hp.sphtfunc.alm2map(omega_tem, fields.nside, lmax=fields.Lmax_map, mmax=fields.Lmax_map)
            tem_dir = f"{fields.dm.cache_dir}/_tems/{deflect_typ}/{exp}" if kappa_rec else f"{fields.dm.cache_dir}/_tems"
            save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError(
            "Must supply arguments: exp Nchi tracer_noise nsims kappa_rec kappa_qe_typ _id")
    exp = str(args[0])
    Nchi = int(args[1])
    tracer_noise = parse_boolean(args[2])
    nsims = int(args[3])
    kappa_rec = parse_boolean(args[4])
    kappa_qe_typ = str(args[5])
    _id = str(args[6])
    main(exp, Nchi, tracer_noise, nsims, kappa_rec, kappa_qe_typ, _id)
