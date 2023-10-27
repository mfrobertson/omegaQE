from DEMNUnii.fields import Fields
from omegaqe.tools import mpi, parse_boolean, none_or_str
import sys
import os


def save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ, gauss_lss, len_lss, sht):
    if not os.path.exists(tem_dir):
        os.makedirs(tem_dir)
    filename = f"omega_tem_{sim}"
    extension = "_wN" if tracer_noise else ""
    if kappa_rec:
        extension += f"_{kappa_qe_typ}"
    if gauss_lss:
        extension += "_gauss"
    elif len_lss:
        extension += "_len"
    full_path = f"{tem_dir}/{filename}{extension}.fits"
    mpi.output(f"Saving template to {full_path}", 0, _id)
    sht.write_map(f"{full_path}", tem_map)


def main(exp, field_typs, Nchi, tracer_noise, start, end, kappa_rec, kappa_qe_typ, deflect_typ, gauss_lss, len_lss, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, fields: {field_typs}, Nchi: {Nchi}, start: {start}, end: {end}, tracer_noise: {tracer_noise}, kappa_rec: {kappa_rec}, kappa_qe_typ: {kappa_qe_typ}, deflect_typ: {deflect_typ}, gauss_lss: {gauss_lss}, len_lss: {len_lss}, nthreads: {nthreads}", 0, _id)

    deflect_typs = ["pbdem_dem", "pbdem_zero", "npbdem_dem", "zero_dem"] if deflect_typ is None else [deflect_typ]
    for sim in range(start, end):
        mpi.output(f"Sim: {sim}", 0, _id)
        fields = Fields(exp, field_typs, use_lss_cache=True, use_cmb_cache=True, cmb_sim=sim, deflect_typ=deflect_typ, gauss_lss=gauss_lss, len_lss=len_lss, nthreads=nthreads)
        for deflect_typ in deflect_typs:
            fields.deflect_typ = deflect_typ
            mpi.output(f"  type: {deflect_typ}", 0, _id)
            neg_tracers = True if deflect_typ=="npbdem_dem" else False
            omega_tem = fields.omega_template(Nchi, tracer_noise=tracer_noise, use_kappa_rec=kappa_rec, kappa_rec_qe_typ=kappa_qe_typ, neg_tracers=neg_tracers)
            tem_map = fields.dm.sht.alm2map(omega_tem, nthreads=fields.nthreads)
            tem_dir = f"{fields.dm.cache_dir}/_tems/{deflect_typ}/{exp}/{field_typs}"
            save_tem_map(tem_map, tem_dir, sim, tracer_noise, kappa_rec, kappa_qe_typ, gauss_lss, len_lss, fields.dm.sht)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 13:
        raise ValueError(
            "Must supply arguments: exp fields Nchi tracer_noise start end kappa_rec kappa_qe_typ deflect_typ gauss_lss len_lss nthreads _id")
    exp = str(args[0])
    fields = str(args[1])
    Nchi = int(args[2])
    tracer_noise = parse_boolean(args[3])
    start = int(args[4])
    end = int(args[5])
    kappa_rec = parse_boolean(args[6])
    kappa_qe_typ = str(args[7])
    deflect_typ = none_or_str(args[8])
    gauss_lss = parse_boolean(args[9])
    len_lss = parse_boolean(args[10])
    nthreads = int(args[11])
    _id = str(args[12])
    main(exp, fields, Nchi, tracer_noise, start, end, kappa_rec, kappa_qe_typ, deflect_typ, gauss_lss, len_lss, nthreads, _id)
