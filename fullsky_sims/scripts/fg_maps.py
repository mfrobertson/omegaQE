import numpy as np
from fullsky_sims.fields import Fields
import sys
import os
from omegaqe.tools import mpi, none_or_str, parse_boolean
import fullsky_sims


def setup_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

def main(start, end, deflect_typ, nu, tsz, ksz, cib, rad, gauss, nbody_label, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"start: {start}, end: {end}, nu: {nu},  tsz: {tsz}, ksz: {ksz}, cib: {cib}, rad: {rad}, gauss: {gauss}, nbody: {nbody_label}, nthreads: {nthreads}", 0, _id)

    dir_name = f"foreground_maps_{nu}"
    if gauss: dir_name += "_gauss"
    nbody = fullsky_sims.wrapper_class(nbody_label, nthreads)
    npix = nbody.sht.nside2npix(nbody.nside)
    T_fg = np.zeros(npix)
    Q_fg = np.zeros(npix)
    U_fg = np.zeros(npix)

    T_tsz = nbody.get_obs_tsz_map(nu)
    T_ksz = nbody.get_obs_ksz_map()
    T_cib = nbody.get_obs_cib_map(nu, muK=True)
    T_rad, Q_rad, U_rad = nbody.get_obs_rad_maps(nu)
    
    if tsz:
        T_fg += T_tsz
        dir_name += "_tsz"
    if ksz:
        T_fg += T_ksz
        dir_name += "_ksz"
    if cib:
        T_fg += T_cib
        dir_name += "_cib"
    if rad:
        T_fg += T_rad
        # Q_fg += Q_rad
        # U_fg += U_rad
        dir_name += "_rad"

    full_dir = f"{nbody.sims_dir}/{deflect_typ}/{dir_name}"
    setup_dir(full_dir)

    for sim in np.arange(start, end):
        mpi.output(f"Sim: {sim}", 0, _id)
        T, Q, U = nbody.sht.read_map(f"{nbody.sims_dir}/{deflect_typ}/TQU_{sim}.fits")
        if gauss:
            cl_T_fg = nbody.sht.map2cl(T_fg)
            fg_lm_gauss = nbody.sht.synalm(cl_T_fg, nbody.sht.lmax)
            T += nbody.sht.alm2map(fg_lm_gauss)
        else:           
            T += T_fg
            # Q += Q_fg
            # U += U_fg
        nbody.sht.write_map(f"{full_dir}/TQU_{sim}.fits", (T, Q, U))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 12:
        raise ValueError(
            "Must supply arguments: start end deflect_typ nu tsz ksz cib rad gauss nbody_label nthreads _id")
    start = int(args[0])
    end = int(args[1])
    deflect_typ = none_or_str(args[2])
    nu = int(args[3])
    tsz = parse_boolean(args[4])
    ksz = parse_boolean(args[5])
    cib = parse_boolean(args[6])
    rad = parse_boolean(args[7])
    gauss = parse_boolean(args[8])
    nbody_label = str(args[9])
    nthreads  = int(args[10])
    _id = str(args[11])
    main(start, end, deflect_typ, nu, tsz, ksz, cib, rad, gauss, nbody_label, nthreads, _id)
