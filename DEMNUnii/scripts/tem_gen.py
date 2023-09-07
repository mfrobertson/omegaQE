from DEMNUnii.fields import Fields
from omegaqe.tools import mpi, parse_boolean, none_or_str
import sys
import os


def save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ, gauss_lss, sht):
    if not os.path.exists(tem_dir):
        os.makedirs(tem_dir)
    filename = f"omega_tem_{sim}"
    extension = "_wN" if tracer_noise else ""
    if kappa_rec:
        extension += f"_{kappa_qe_typ}"
    if gauss_lss:
        extension += "_gauss"
    full_path = f"{tem_dir}/{filename}{extension}.fits"
    mpi.output(f"Saving template to {full_path}", 0, _id)
    sht.write_map(f"{full_path}", tem_map)


def main(exp, Nchi, tracer_noise, nsims, kappa_rec, kappa_qe_typ, deflect_typ, gauss_lss, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, Nchi: {Nchi}, nsim: {nsims}, tracer_noise: {tracer_noise}, kappa_rec: {kappa_rec}, kappa_qe_typ: {kappa_qe_typ}, deflect_typ: {deflect_typ}, gauss_lss: {gauss_lss}, nthreads: {nthreads}", 0, _id)

    deflect_typs = ["pbdem_dem", "pbdem_zero", "npbdem_dem", "diff_diff", "dem_dem"] if deflect_typ is None else [deflect_typ]
    for sim in range(nsims):
        mpi.output(f"Sim: {sim}", 0, _id)
        fields = Fields(exp, use_lss_cache=True, use_cmb_cache=True, cmb_sim=sim, deflect_typ=deflect_typ, gauss_lss=gauss_lss, nthreads=nthreads)
        for deflect_typ in deflect_typs:
            fields.deflect_typ = deflect_typ
            mpi.output(f"  type: {deflect_typ}", 0, _id)
            neg_tracers = True if deflect_typ=="npbdem_dem" else False
            omega_tem = fields.omega_template(Nchi, tracer_noise=tracer_noise, use_kappa_rec=kappa_rec, kappa_rec_qe_typ=kappa_qe_typ, neg_tracers=neg_tracers)
            tem_map = fields.dm.sht.alm2map(omega_tem, nthreads=fields.nthreads)
            tem_dir = f"{fields.dm.cache_dir}/_tems/{deflect_typ}/{exp}"
            save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ, gauss_lss, fields.dm.sht)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 10:
        raise ValueError(
            "Must supply arguments: exp Nchi tracer_noise nsims kappa_rec kappa_qe_typ deflect_typ gauss_lss, nthreads _id")
    exp = str(args[0])
    Nchi = int(args[1])
    tracer_noise = parse_boolean(args[2])
    nsims = int(args[3])
    kappa_rec = parse_boolean(args[4])
    kappa_qe_typ = str(args[5])
    deflect_typ = none_or_str(args[6])
    gauss_lss = parse_boolean(args[7])
    nthreads = int(args[8])
    _id = str(args[9])
    main(exp, Nchi, tracer_noise, nsims, kappa_rec, kappa_qe_typ, deflect_typ, gauss_lss, nthreads, _id)
