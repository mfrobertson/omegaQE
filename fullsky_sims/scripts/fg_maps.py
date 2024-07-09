import numpy as np
from fullsky_sims.fields import Fields
import sys
import os
from omegaqe.tools import mpi, none_or_str, parse_boolean
import fullsky_sims


def setup_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

def main(start, end, deflect_typ, nu, tsz, ksz, cib, rad, gauss, cluster_mask, point_mask, nbody_label, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"start: {start}, end: {end}, nu: {nu},  tsz: {tsz}, ksz: {ksz}, cib: {cib}, rad: {rad}, gauss: {gauss}, nbody: {nbody_label}, nthreads: {nthreads}", 0, _id)

    dir_name = f"foreground_maps_{nu}"
    if gauss: dir_name += "_gauss"
    nbody = fullsky_sims.wrapper_class(nbody_label, nthreads)
    npix = nbody.sht.nside2npix(nbody.nside)
    T_fg = np.zeros(npix)
    Q_fg = np.zeros(npix)
    U_fg = np.zeros(npix)
    
    if tsz:
        dir_name += "_tsz"
    if ksz:
        dir_name += "_ksz"
    if cib:
        dir_name += "_cib"
    if rad:
        dir_name += "_rad"
    if point_mask:
        dir_name += "_pm"
    if cluster_mask:
        dir_name += "_cm"


    full_dir = f"{nbody.sims_dir}/{deflect_typ}/{dir_name}"
    setup_dir(full_dir)

    for sim in np.arange(start, end):
        mpi.output(f"Sim: {sim}", 0, _id)
        T, Q, U = nbody.sht.read_map(f"{nbody.sims_dir}/{deflect_typ}/TQU_{sim}.fits")
        if gauss:
            input_kappa = nbody.sht.read_map(f"{nbody.sims_dir}//kappa_diff_{sim}.fits")
            T_fg, tracers = nbody.create_gauss_fg_maps(nu, tsz, ksz, cib, rad, point_mask, cluster_mask, return_tracers=True, input_kappa=input_kappa)
            T += T_fg
            nbody.sht.write_map(f"{full_dir}/kgI_{sim}.fits", tracers)
        else:       
            T_fg, Q_fg, U_fg = nbody.create_fg_maps(nu, tsz, ksz, cib, rad, point_mask, cluster_mask)    
            T += T_fg
            Q += Q_fg
            U += U_fg
        nbody.sht.write_map(f"{full_dir}/TQU_{sim}.fits", (T, Q, U))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 14:
        raise ValueError(
            "Must supply arguments: start end deflect_typ nu tsz ksz cib rad gauss cluster point nbody_label nthreads _id")
    start = int(args[0])
    end = int(args[1])
    deflect_typ = none_or_str(args[2])
    nu = int(args[3])
    tsz = parse_boolean(args[4])
    ksz = parse_boolean(args[5])
    cib = parse_boolean(args[6])
    rad = parse_boolean(args[7])
    gauss = parse_boolean(args[8])
    cluster_mask = parse_boolean(args[9])
    point_mask = parse_boolean(args[10])
    nbody_label = str(args[11])
    nthreads  = int(args[12])
    _id = str(args[13])
    main(start, end, deflect_typ, nu, tsz, ksz, cib, rad, gauss, cluster_mask, point_mask, nbody_label, nthreads, _id)
