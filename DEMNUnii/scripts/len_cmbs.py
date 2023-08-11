from omegaqe.powerspectra import Powerspectra
from omegaqe.tools import getFileSep
from DEMNUnii.demnunii import Demnunii
from scipy.interpolate import InterpolatedUnivariateSpline
from plancklens.sims import cmbs, phas
import numpy as np
import sys
import os

power = Powerspectra()
dm = Demnunii()
power.cosmo = dm.cosmo
sep = getFileSep()
# geom = lenspyx.get_geom(('healpix', {'nside': dm.nside}))
LMAX_MAP = 6000

if not 'PLENS' in os.environ.keys():
    os.environ['PLENS'] = '_tmp'


def _raise_deflection_error(deflect_typ):
    raise ValueError(f"Deflection type {deflect_typ} not of demnunii, diff_alpha or diff_omega.")


def _lensing_fac(ells):
    fac = np.zeros(np.size(ells) + 1)
    fac[1:] = -2 / ells / (ells + 1)
    return fac


def _get_plm(nthreads, deflect_typ):
    ells = np.arange(1, LMAX_MAP + 1)
    lensing_fac = _lensing_fac(ells)
    if deflect_typ == "demnunii" or deflect_typ == "diff_omega":
        kappa_map = dm.get_kappa_map()
        # klm = hp.sphtfunc.map2alm(kappa_map, lmax, lmax, use_pixel_weights=True)
        # klm = geom.map2alm(kappa_map, LMAX_MAP, LMAX_MAP, nthreads)
        # return almxfl(klm, lensing_fac, None, False)
        klm = dm.sht.map2alm(kappa_map, lmax=LMAX_MAP, nthread=nthreads)
        return dm.sht.almxfl(klm, lensing_fac)
    if deflect_typ == "diff_alpha":
        Cl_phi = np.zeros(np.size(ells) + 1)
        Cl_phi[1:] = power.get_phi_ps(ells, extended=True)
        # return synalm(Cl_phi, lmax=LMAX_MAP, mmax=LMAX_MAP)
        return dm.sht.synalm(Cl_phi, lmax=LMAX_MAP)
    _raise_deflection_error(deflect_typ)


def _get_Olm(nthreads, deflect_typ, camb_acc):
    ells = np.arange(1, LMAX_MAP + 1)
    lensing_fac = _lensing_fac(ells)
    if deflect_typ == "demnunii":
        omega_map = dm.get_omega_map()
        # olm = hp.sphtfunc.map2alm(omega_map, lmax, lmax, use_pixel_weights=True)
        # olm = geom.map2alm(omega_map, LMAX_MAP, LMAX_MAP, nthreads)
        # return almxfl(olm, lensing_fac, None, False)
        olm = dm.sht.map2alm(omega_map, lmax=LMAX_MAP, nthreads=nthreads)
        return dm.sht.almxfl(olm, lensing_fac)
    if deflect_typ == "diff_omega" or "diff_alpha":
        ells_omega, omega_ps = dm.cosmo.get_postborn_omega_ps(acc=camb_acc)
        Cl_omega = np.zeros(np.size(ells) + 1)
        Cl_omega[1:] = InterpolatedUnivariateSpline(ells_omega, omega_ps)(ells)
        # olm = synalm(Cl_omega, lmax=LMAX_MAP, mmax=LMAX_MAP)
        # return almxfl(olm, lensing_fac, None, False)
        olm = dm.sht.synalm(Cl_omega, lmax=LMAX_MAP)
        return dm.sht.almxfl(olm, lensing_fac)
    _raise_deflection_error(deflect_typ)


def get_deflection_fields(nthreads, deflect_typ="demnunii", camb_acc=4):
    plm = _get_plm(nthreads, deflect_typ)
    Olm = _get_Olm(nthreads, deflect_typ, camb_acc)
    # glm = almxfl(plm, np.sqrt(np.arange(LMAX_MAP + 1, dtype=float) * np.arange(1, LMAX_MAP + 2)), None, False)
    # clm = almxfl(Olm, np.sqrt(np.arange(LMAX_MAP + 1, dtype=float) * np.arange(1, LMAX_MAP + 2)), None, False)
    ells = np.arange(1, LMAX_MAP + 1)
    lensing_fac = _lensing_fac(ells)
    glm = dm.sht.almxfl(plm, np.sqrt(lensing_fac))
    clm = dm.sht.almxfl(Olm, np.sqrt(lensing_fac))
    return glm, clm


def get_unlensed_cmb_ps():
    Tcmb = 2.7255
    indices = ["TT", "EE", "BB", "TE"]
    return {idx.lower(): dm.cosmo.get_unlens_ps(idx, ellmax=LMAX_MAP)[:LMAX_MAP + 1] * (Tcmb * 1e6) ** 2 for idx in indices}


def get_unlensed_alms(unl_cmb_spectra, sim):
    lib_pha = phas.lib_phas(os.path.join(os.environ['PLENS'], 'len_cmbs', 'phas'), 3, LMAX_MAP)
    unl_lib = cmbs.sims_cmb_unl(unl_cmb_spectra, lib_pha)
    return unl_lib.get_sim_tlm(sim), unl_lib.get_sim_elm(sim), unl_lib.get_sim_blm(sim)


def get_lensed_maps(dlm, unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    Elm_unl = unl_alms[1]
    Blm_unl = unl_alms[2]
    # geom_info = ('healpix', {'nside': dm.nside})
    # return lenspyx.alm2lenmap([Tlm_unl, Elm_unl, Blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=1e-10, nthreads=nthreads)
    return dm.sht.alm2lenmap([Tlm_unl, Elm_unl, Blm_unl], dlm, nthreads=nthreads)

def get_unlensed_Tmap(unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    # return lenspyx.get_geom(('healpix', {'nside': demnunii.nside})).alm2map(Tlm_unl, LMAX_MAP, LMAX_MAP, nthreads=nthreads)
    return dm.sht.alm2map(Tlm_unl, lmax=LMAX_MAP, nthreads=nthreads)


def save_lens_maps(loc, len_maps, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # hp.fitsfunc.write_map(f"{directory}{sep}TQU_{sim}.fits", len_maps, dtype=float, overwrite=True)
    dm.sht.write_map(f"{directory}{sep}TQU_{sim}.fits", len_maps)


def save_omega(loc, clm, nthreads):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    ells = np.arange(np.size(clm))
    # olm = almxfl(clm, -np.sqrt(ells * (ells+1))/2, None, False)
    # omega_map = geom.alm2map(olm, LMAX_MAP, LMAX_MAP, nthreads)
    # hp.fitsfunc.write_map(f"{directory}{sep}omega_true.fits", omega_map, dtype=float, overwrite=True)
    olm = dm.sht.almxfl(clm, -np.sqrt(ells * (ells + 1)) / 2)
    omega_map = dm.sht.alm2map(olm, lmax=LMAX_MAP, nthreads=nthreads)
    dm.sht.write_map(f"{directory}{sep}omega_true.fits", omega_map)


def save_omega_kappa(loc, dlm, nthreads):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    alm = dlm[0]
    ells = np.arange(np.size(alm))
    # klm = almxfl(alm, -np.sqrt(ells * (ells + 1)) / 2, None, False)
    # kappa_map = geom.alm2map(klm, LMAX_MAP, LMAX_MAP, nthreads)
    # hp.fitsfunc.write_map(f"{directory}{sep}kappa_true.fits", kappa_map, dtype=float, overwrite=True)
    klm = dm.sht.almxfl(alm, -np.sqrt(ells * (ells + 1)) / 2)
    kappa_map = dm.sht.alm2map(klm, lmax=LMAX_MAP, nthreads=nthreads)
    dm.sht.write_map(f"{directory}{sep}kappa_true.fits", kappa_map)
    save_omega(loc, dlm[1], nthreads)


def save_Tunl(loc, Tmap, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # hp.fitsfunc.write_map(f"{directory}{sep}T_{sim}.fits", Tmap, dtype=float, overwrite=True)
    dm.sht.write_map(f"{directory}{sep}T_{sim}.fits", Tmap)

def main(nsims, nthreads, loc):
    glm, clm = get_deflection_fields(nthreads, deflect_typ="demnunii")
    demnunii_dir = f"{loc}{sep}demnunii"
    dlm_dem = np.array([glm, clm])

    glm, clm = get_deflection_fields(nthreads, deflect_typ="diff_alpha", camb_acc=4)
    diff_alpha_dir = f"{loc}{sep}diff_alpha"
    dlm_diff_alpha = np.array([glm, clm])
    save_omega_kappa(diff_alpha_dir, dlm_diff_alpha, nthreads)

    glm, clm = get_deflection_fields(nthreads, deflect_typ="diff_omega", camb_acc=4)
    diff_omega_dir = f"{loc}{sep}diff_omega"
    save_omega(diff_omega_dir, clm, nthreads)
    dlm_diff_omega = np.array([glm, clm])

    unl_cmb_spectra = get_unlensed_cmb_ps()
    for sim in range(nsims):
        unl_alms = get_unlensed_alms(unl_cmb_spectra, sim)
        len_maps_dem = get_lensed_maps(dlm_dem, unl_alms, nthreads)
        save_lens_maps(demnunii_dir, len_maps_dem, sim)

        len_maps_diff_alpha = get_lensed_maps(dlm_diff_alpha, unl_alms, nthreads)
        save_lens_maps(diff_alpha_dir, len_maps_diff_alpha, sim)

        len_maps_diff_omega = get_lensed_maps(dlm_diff_omega, unl_alms, nthreads)
        save_lens_maps(diff_omega_dir, len_maps_diff_omega, sim)

        save_Tunl(f"{loc}{sep}T_unl", get_unlensed_Tmap(unl_alms, nthreads), sim)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) !=3:
        raise ValueError(
            "Arguments should be nsims nthreads loc")
    nsims = int(args[0])
    nthreads = int(args[1])
    loc = str(args[2])
    main(nsims, nthreads, loc)
