import lenspyx
from lenspyx.utils_hp import synalm, almxfl
from omegaqe.powerspectra import Powerspectra
from omegaqe.tools import getFileSep
from demnunii import Demnunii
from scipy.interpolate import InterpolatedUnivariateSpline
from plancklens.sims import cmbs, phas
import healpy as hp
import numpy as np
import sys
import os

power = Powerspectra()
demnunii = Demnunii()
power.cosmo = demnunii.cosmo
sep = getFileSep()

LMAX_MAP = 6000

if not 'PLENS' in os.environ.keys():
    os.environ['PLENS'] = '_tmp'


def get_deflection_fields_demnunii(lmax):
    kappa_map = demnunii.get_kappa_map()
    omega_map = demnunii.get_omega_map()
    klm = hp.sphtfunc.map2alm(kappa_map, lmax, lmax, use_pixel_weights=True)
    olm = hp.sphtfunc.map2alm(omega_map, lmax, lmax, use_pixel_weights=True)

    ells = np.arange(1, lmax + 1)
    fac = np.zeros(np.size(ells) + 1)
    fac[1:] = 2 / ells / (ells + 1)
    plm = hp.sphtfunc.almxfl(klm, fac, None, False)
    Olm = hp.sphtfunc.almxfl(olm, fac, None, False)

    glm = hp.sphtfunc.almxfl(plm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)
    clm = hp.sphtfunc.almxfl(Olm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)

    return glm, clm


def get_deflection_fields_lenspyx(lmax, omega_acc=4):
    ells = np.arange(1, lmax + 1)
    Cl_phi = np.zeros(np.size(ells) + 1)
    Cl_phi[1:] = power.get_phi_ps(ells)
    plm = synalm(Cl_phi, lmax=lmax, mmax=lmax)

    ells_omega, omega_ps = demnunii.cosmo.get_postborn_omega_ps(acc=omega_acc)
    Cl_omega = np.zeros(np.size(ells) + 1)
    Cl_omega[1:] = InterpolatedUnivariateSpline(ells_omega, omega_ps)(ells)
    olm = synalm(Cl_omega, lmax=lmax, mmax=lmax)
    fac = np.zeros(np.size(ells) + 1)
    fac[1:] = 2 / ells / (ells + 1)
    Olm = almxfl(olm, fac, None, False)

    glm = hp.sphtfunc.almxfl(plm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)
    clm = hp.sphtfunc.almxfl(Olm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)

    return glm, clm


def get_unlensed_cmb_ps(lmax):
    Tcmb = 2.7255
    indices = ["TT", "EE", "BB", "TE"]
    return {idx.lower(): demnunii.cosmo.get_unlens_ps(idx, ellmax=lmax)[:lmax + 1] * (Tcmb * 1e6) ** 2 for idx in indices}


def get_unlensed_alms(lmax, unl_cmb_spectra):
    print("new_unl_alms")
    lib_pha = phas.lib_phas(os.path.join(os.environ['PLENS'], 'len_cmbs', 'phas'), 3, lmax)
    print(unl_cmb_spectra.keys())
    unl_lib = cmbs.sims_cmb_unl(unl_cmb_spectra, lib_pha)
    return unl_lib.get_sim_tlm(0), unl_lib.get_sim_elm(0), unl_lib.get_sim_blm(0)


def get_lensed_maps(dlm, unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    Elm_unl = unl_alms[1]
    Blm_unl = unl_alms[2]
    geom_info = ('healpix', {'nside': demnunii.nside})
    return lenspyx.alm2lenmap([Tlm_unl, Elm_unl, Blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=1e-10, nthreads=nthreads)


def get_unlensed_Tmap(unl_alms, lmax, nthreads):
    Tlm_unl = unl_alms[0]
    return lenspyx.get_geom(('healpix', {'nside': demnunii.nside})).alm2map(Tlm_unl, lmax, lmax, nthreads=nthreads)


def save_lens_maps(loc, len_maps, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    hp.fitsfunc.write_map(f"{directory}{sep}TQU_{sim}.fits", len_maps)


def save_Tunl(loc, Tmap, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    hp.fitsfunc.write_map(f"{directory}{sep}T_{sim}.fits", Tmap)


def main(nsims, nthreads, loc):
    lmax_map=LMAX_MAP
    glm, clm = get_deflection_fields_demnunii(lmax_map)
    dlm_dem = np.array([glm, clm])
    glm, clm = get_deflection_fields_lenspyx(lmax_map)
    dlm_diff_alpha = np.array([glm, clm])
    unl_cmb_spectra = get_unlensed_cmb_ps(lmax_map)
    for sim in range(nsims):
        unl_alms = get_unlensed_alms(lmax_map, unl_cmb_spectra)
        len_maps_dem = get_lensed_maps(dlm_dem, unl_alms, nthreads)
        save_lens_maps(f"{loc}{sep}demnunii", len_maps_dem, sim)
        len_maps_diff_phi = get_lensed_maps(dlm_diff_alpha, unl_alms, nthreads)
        save_lens_maps(f"{loc}{sep}diff_alpha", len_maps_diff_phi, sim)
        save_Tunl(f"{loc}{sep}T_unl", get_unlensed_Tmap(unl_alms, lmax_map, nthreads), sim)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) !=3:
        raise ValueError(
            "Arguments should be nsims nthreads loc")
    nsims = int(args[0])
    nthreads = int(args[1])
    loc = str(args[2])
    main(nsims, nthreads, loc)
