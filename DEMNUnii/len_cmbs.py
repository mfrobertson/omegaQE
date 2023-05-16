import lenspyx
from lenspyx.utils_hp import synalm
from omegaqe.cosmology import Cosmology
from omegaqe.tools import getFileSep
from demnunii import Demnunii
import healpy as hp
import numpy as np
import sys
import os

cosmo = Cosmology("DEMNUnii_params.ini")
demnunii = Demnunii()


def get_deflection_fields(lmax):
    kappa_map = demnunii.get_kappa_map()
    omega_map = demnunii.get_omega_map()
    klm = hp.sphtfunc.map2alm(kappa_map, lmax, lmax, use_pixel_weights=True)
    olm = hp.sphtfunc.map2alm(omega_map, lmax, lmax, use_pixel_weights=True)

    ells = np.arange(lmax + 1)
    fac = 2 / ells / (ells + 1)
    fac[0] = 0
    plm = hp.sphtfunc.almxfl(klm, fac, None, False)
    Olm = hp.sphtfunc.almxfl(olm, fac, None, False)

    glm = hp.sphtfunc.almxfl(plm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)
    clm = hp.sphtfunc.almxfl(Olm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)

    return glm, clm


def get_unlensed_cmb_ps():
    Tcmb = 2.7255
    Cl_TT_unl = cosmo.get_unlens_ps("TT") * (Tcmb * 1e6) ** 2
    Cl_EE_unl = cosmo.get_unlens_ps("EE") * (Tcmb * 1e6) ** 2
    Cl_BB_unl = cosmo.get_unlens_ps("BB") * (Tcmb * 1e6) ** 2
    return Cl_TT_unl, Cl_EE_unl, Cl_BB_unl


def get_unlensed_alms(lmax, unl_cmb_spectra):
    Cl_TT_unl = unl_cmb_spectra[0]
    Cl_EE_unl = unl_cmb_spectra[1]
    Cl_BB_unl = unl_cmb_spectra[2]

    tlm_unl = synalm(Cl_TT_unl, lmax=lmax, mmax=lmax)
    elm_unl = synalm(Cl_EE_unl, lmax=lmax, mmax=lmax)
    blm_unl = synalm(Cl_BB_unl, lmax=lmax, mmax=lmax)

    return tlm_unl, elm_unl, blm_unl


def get_lensed_maps(dlm, unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    Elm_unl = unl_alms[1]
    Blm_unl = unl_alms[2]
    geom_info = ('healpix', {'nside': demnunii.nside})
    Tlen, Qlen, Ulen = lenspyx.alm2lenmap([Tlm_unl, Elm_unl, Blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=1e-6, nthreads=nthreads)
    return Tlen, Qlen, Ulen


def save_maps(loc, len_maps, sim):
    sep = getFileSep()
    directory = f"{loc}{sep}{sim}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    hp.fitsfunc.write_map(f"{directory}{sep}TQU", len_maps)


def main(lmax, nsims, nthreads, loc):
    glm, clm = get_deflection_fields(lmax)
    dlm = np.array([glm, clm])
    Tlm_unl, Elm_unl, Blm_unl = get_unlensed_cmb_ps()
    unl_cmb_spectra = np.array([Tlm_unl, Elm_unl, Blm_unl])
    for sim in range(nsims):
        unl_alms = get_unlensed_alms(lmax, unl_cmb_spectra)
        len_maps = get_lensed_maps(dlm, unl_alms, nthreads)
        save_maps(loc, len_maps, sim)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        raise ValueError(
            "Arguments should be lmax nsims nthreads loc")
    lmax = int(args[0])
    nsims = int(args[1])
    nthreads = int(args[2])
    loc = str(args[3])
    main(lmax, nsims, nthreads, loc)
