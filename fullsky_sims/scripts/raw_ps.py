from omegaqe.tools import mpi, parse_boolean, none_or_str
import sys
import os
import numpy as np
import fullsky_sims


def _save_ps(ps, exp, tracer_fields, nsims, deflect_typ, ext, qe_typ, cmb_noise):
    ps_dir = f"{nbody.cache_dir}/_raw_ps/{deflect_typ}/{exp}/{tracer_fields}"
    if not os.path.exists(ps_dir):
        os.makedirs(ps_dir)
    ps_filename = f"ps"
    extension = f"_{nsims}"
    extension += ext
    extension += f"_{qe_typ}"
    if not cmb_noise:
        extension += "_ncmbN"
    ps_full_path = f"{ps_dir}/{ps_filename}{extension}.npy"
    mpi.output(f"Saving cross-spectrum to {ps_full_path}", 0, _id)
    np.save(f"{ps_full_path}", ps)

def _get_iter_mc_corr_w(exp, qe_typ, gmv):
    offset=10
    nbins=30
    ext = "_gmv" if gmv else ""
    dir_loc = f"{nbody.cache_dir}/_iter_norm/{exp}/{qe_typ}/{offset}_{nbins}"
    return np.load(f"{dir_loc}/iter_norm_w{ext}.npy")


def _get_ps(exp, tracer_fields, deflect_typ, tem_ext, nsims, iter_mc_corr, cmb_noise, gmv, bh, nthreads, qe_typ):
    mpi.output(f"  sim: 0", 0, _id)
    nbody.sht.nthreads = nthreads
    if not cmb_noise:
        ext = "_nN"
        if gmv:
            ext += "_gmv"
        if bh:
            ext += "_bh"            
    elif gmv:
        ext = "__gmv"
        if bh:
            ext += "_bh" 
    elif bh:
        ext = "_bh"
    else:
        ext = "_"

    omega_tem = nbody.sht.read_map(f"{nbody.cache_dir}/_tems/{deflect_typ}/{exp}/{tracer_fields}/omega_tem_{0}{tem_ext}.fits")
    omega_rec = nbody.sht.read_map(f"{nbody.sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_{0}{ext}.fits")
    if iter_mc_corr:
        mc_corr = _get_iter_mc_corr_w(exp, qe_typ, gmv)
        omega_rec = nbody.sht.alm2map(nbody.sht.almxfl(nbody.sht.map2alm(omega_rec), 1/mc_corr))
    Cl_ww = nbody.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
    for sim in range(1,nsims):
        mpi.output(f"  sim: {sim}", 0, _id)
        omega_tem = nbody.sht.read_map(f"{nbody.cache_dir}/_tems/{deflect_typ}/{exp}/{tracer_fields}/omega_tem_{sim}{tem_ext}.fits")
        omega_rec = nbody.sht.read_map(f"{nbody.sims_dir}/{deflect_typ}/{exp}/omega/{qe_typ}_{sim}{ext}.fits")
        if iter_mc_corr:
            omega_rec = nbody.sht.alm2map(nbody.sht.almxfl(nbody.sht.map2alm(omega_rec), 1/mc_corr))
        Cl_ww += nbody.sht.map2cl(omega_tem, omega_rec, nthreads=nthreads)
    return Cl_ww/nsims


def main(exp, tracer_fields, tracer_noise, kappa_rec, qe_typ, nsims, deflect_typ, gauss_lss, len_lss, iter_mc_corr, cmb_noise, gmv, bh, nbody_name, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"exp: {exp}, tracer_fields: {tracer_fields}, nsims: {nsims}, deflect_typ: {deflect_typ}, gauss_lss: {gauss_lss}, len_lss: {len_lss}, cmb_noise: {cmb_noise}, gmv: {gmv}, bh: {bh}, nthreads: {nthreads}", 0, _id)

    global nbody
    nbody = fullsky_sims.wrapper_class(nbody_name, nthreads)

    deflect_typs = ["pbdem_dem", "pbdem_zero"] if deflect_typ is None else [deflect_typ]

    ext = "_wN" if tracer_noise else ""
    if kappa_rec:
        ext += f"_{qe_typ}"
    if gauss_lss:
        ext += "_gauss"
    elif len_lss:
        ext += "_len"
    if iter_mc_corr:
        ext += "_mc"
    if not cmb_noise:
        ext += "_nN"
    if gmv:
        ext += "_gmv"
    if bh:
        ext += "_bh"
    for deflect_typ in deflect_typs:
        mpi.output(f"Type: {deflect_typ}", 0, _id)
        ps = _get_ps(exp, tracer_fields, deflect_typ, ext, nsims, iter_mc_corr, cmb_noise, gmv, bh, nthreads, qe_typ)
        _save_ps(ps, exp, tracer_fields, nsims, deflect_typ, ext, qe_typ, cmb_noise)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 16:
        raise ValueError(
            "Must supply arguments: exp tracer_fields tracer_noise kappa_rec qe_typ nsims deflect_typ gauss_lss len_lss iter_mc_corr cmb_noise gmv bh nbody nthreads _id")
    exp = str(args[0])
    tracer_fields = str(args[1])
    tracer_noise = parse_boolean(args[2])
    kappa_rec = parse_boolean(args[3])
    qe_typ = str(args[4])
    nsims = int(args[5])
    deflect_typ = none_or_str(args[6])
    gauss_lss = parse_boolean(args[7])
    len_lss = parse_boolean(args[8])
    iter_mc_corr = parse_boolean(args[9])
    cmb_noise = parse_boolean(args[10])
    gmv = parse_boolean(args[11])
    bh = parse_boolean(args[12])
    nbody_name = str(args[13])
    nthreads = int(args[14])
    _id = str(args[15])
    main(exp, tracer_fields, tracer_noise, kappa_rec, qe_typ, nsims, deflect_typ, gauss_lss, len_lss, iter_mc_corr, cmb_noise, gmv, bh, nbody_name, nthreads, _id)
