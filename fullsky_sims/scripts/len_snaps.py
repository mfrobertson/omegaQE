from omegaqe.tools import getFileSep
from fullsky_sims.demnunii import Demnunii
import fullsky_sims
import numpy as np
import sys
import os

dm = Demnunii()
sep = getFileSep()
LMAX_MAP = 6000

if not 'PLENS' in os.environ.keys():
    os.environ['PLENS'] = '_tmp'


def _lensing_fac():
    ells = np.arange(LMAX_MAP+1)[1:]
    fac = np.zeros(LMAX_MAP+1)
    fac[1:] = 2 / np.sqrt(ells * (ells + 1))
    return fac


def get_glm(nthreads, snap):
    lensing_fac = _lensing_fac()
    gal_kappa_map = dm.get_gal_kappa_map(snap)
    klm = dm.sht.map2alm(gal_kappa_map, lmax=LMAX_MAP, nthreads=nthreads)
    return dm.sht.almxfl(klm, lensing_fac)


def get_lensed_map(dlm, unl_alm, nthreads):
    return dm.sht.alm2lenmap(unl_alm, dlm, nthreads=nthreads)


def save_lens_maps(len_map, snap):
    directory = f"{dm.cache_dir}{sep}_len_snaps"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    snap_num_app = "0" if len(str(snap)) == 2 else "00"
    dm.sht.write_map(f"{directory}{sep}len_snap_{snap_num_app}{snap}.fits", len_map)


def main(nthreads):
    for snap in np.arange(63):
        
        glm_snap = get_glm(nthreads, snap)
        unl_snap = dm.get_particle_snap(snap)
        unl_alm = dm.sht.map2alm(unl_snap, lmax=LMAX_MAP, nthreads=nthreads)
        len_map = get_lensed_map(glm_snap, unl_alm, nthreads)
        save_lens_maps(len_map, snap)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError(
            "Arguments should be nthreads")
    nthreads = int(args[0])
    main(nthreads)
