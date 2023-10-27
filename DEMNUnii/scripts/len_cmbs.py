from omegaqe.tools import getFileSep
from DEMNUnii.demnunii import Demnunii
from plancklens.sims import cmbs, phas
import numpy as np
import sys
import os

dm = Demnunii()
sep = getFileSep()
LMAX_MAP = 6000

if not 'PLENS' in os.environ.keys():
    os.environ['PLENS'] = '_tmp'


def _check_deflect_typ(deflect_typ, curl):
    if curl:
        if deflect_typ not in ["dem", "diff"]:
            _raise_deflection_error(deflect_typ)
    if deflect_typ not in ["dem","diff","pb"]:
        _raise_deflection_error(deflect_typ)


def _raise_deflection_error(deflect_typ):
    raise ValueError(f"Deflection type {deflect_typ} not of dem, diff (or pb for gradient only).")


def _lensing_fac():
    ells = np.arange(LMAX_MAP+1)[1:]
    fac = np.zeros(LMAX_MAP+1)
    fac[1:] = -2 / np.sqrt(ells * (ells + 1))
    return fac


def _inv_lensing_fac():
    inv_fac = np.zeros(LMAX_MAP+1)
    inv_fac[1:] = 1/_lensing_fac()[1:]
    return inv_fac


def get_glm(nthreads, deflect_typ):
    _check_deflect_typ(deflect_typ, curl=False)
    lensing_fac = _lensing_fac()
    klm = None
    if deflect_typ == "dem":
        kappa_map = dm.get_kappa_map()
        klm = dm.sht.map2alm(kappa_map, lmax=LMAX_MAP, nthreads=nthreads)
    elif deflect_typ == "pb":
        kappa_map = dm.get_kappa_map(pb=True)
        klm = dm.sht.map2alm(kappa_map, lmax=LMAX_MAP, nthreads=nthreads)
    elif deflect_typ == "diff":
        kappa_map = dm.get_kappa_map(pb=True)
        Cl_kappa = dm.sht.map2cl(kappa_map, lmax=LMAX_MAP, lmax_out=LMAX_MAP, nthreads=nthreads)
        Cl_kappa_smooth = dm.sht.smoothed_cl(Cl_kappa, nbins=150, zerod=False)
        klm = dm.sht.synalm(Cl_kappa_smooth, lmax=LMAX_MAP)
    return dm.sht.almxfl(klm, lensing_fac)



def get_clm(nthreads, deflect_typ):
    _check_deflect_typ(deflect_typ, curl=True)
    lensing_fac = _lensing_fac()
    olm = None
    omega_map = dm.get_omega_map()
    if deflect_typ == "dem":
        olm = dm.sht.map2alm(omega_map, lmax=LMAX_MAP, nthreads=nthreads)
    elif deflect_typ == "diff":
        Cl_omega_smooth = dm.sht.map2cl(omega_map, lmax=LMAX_MAP, lmax_out=LMAX_MAP, nthreads=nthreads, smoothing_nbins=150)
        olm = dm.sht.synalm(Cl_omega_smooth, lmax=LMAX_MAP)
    return dm.sht.almxfl(olm, lensing_fac)


def get_unlensed_cmb_ps():
    Tcmb = 2.7255
    indices = ["TT", "EE", "BB", "TE"]
    return {idx.lower(): dm.cosmo.get_unlens_ps(idx, ellmax=LMAX_MAP)[:LMAX_MAP + 1] * (Tcmb * 1e6) ** 2 for idx in indices}


def get_unlensed_alms(unl_cmb_spectra, sim, unl_loc):
    if unl_loc is None:
        lib_pha = phas.lib_phas(os.path.join(os.environ['PLENS'], 'len_cmbs', 'phas'), 3, LMAX_MAP)
        unl_lib = cmbs.sims_cmb_unl(unl_cmb_spectra, lib_pha)
        return unl_lib.get_sim_tlm(sim), unl_lib.get_sim_elm(sim), unl_lib.get_sim_blm(sim)
    TQU = dm.sht.read_map(f"unl_loc{sep}TQU_{sim}.fits")
    T_unl = dm.sht.map2alm(TQU[0])
    E_unl, B_unl = dm.sht.map2alm_spin(TQU[1:], 2)
    return T_unl, E_unl, B_unl


def get_lensed_maps(dlm, unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    Elm_unl = unl_alms[1]
    Blm_unl = unl_alms[2]
    return dm.sht.alm2lenmap([Tlm_unl, Elm_unl, Blm_unl], dlm, nthreads=nthreads)

def get_unlensed_Tmap(unl_alms, nthreads):
    Tlm_unl = unl_alms[0]
    return dm.sht.alm2map(Tlm_unl, lmax=LMAX_MAP, nthreads=nthreads)


def save_lens_maps(loc, len_maps, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    dm.sht.write_map(f"{directory}{sep}TQU_{sim}.fits", len_maps)


def _save_omega_diff(loc, clm, nthreads):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    olm = dm.sht.almxfl(clm, _inv_lensing_fac())
    omega_map = dm.sht.alm2map(olm, lmax=LMAX_MAP, nthreads=nthreads)
    dm.sht.write_map(f"{directory}{sep}omega_diff.fits", omega_map)


def _save_kappa_diff(loc, glm, nthreads):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    klm = dm.sht.almxfl(glm, _inv_lensing_fac())
    kappa_map = dm.sht.alm2map(klm, lmax=LMAX_MAP, nthreads=nthreads)
    dm.sht.write_map(f"{directory}{sep}kappa_diff.fits", kappa_map)


def save_Tunl(loc, Tmap, sim):
    directory = f"{loc}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    dm.sht.write_map(f"{directory}{sep}T_{sim}.fits", Tmap)

def _save_unl_cmbs(loc, alms, sim, nthreads):
    directory = f"{loc}{sep}unlensed"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    Tmap = dm.sht.alm2map(alms[0], lmax=LMAX_MAP, nthreads=nthreads)
    QUmaps = dm.sht.alm2map_spin(alms[1:], 2, lmax=LMAX_MAP, nthreads=nthreads)
    dm.sht.write_map(f"{directory}{sep}TQU_{sim}.fits", np.array([Tmap, QUmaps[0], QUmaps[1]]))


def main(nsims, nthreads, loc, unl_loc):
    glm_pb = get_glm(nthreads, "pb")
    glm_dem = get_glm(nthreads, "dem")
    glm_diff = get_glm(nthreads, "diff")
    clm_dem = get_clm(nthreads, "dem")
    clm_diff = get_clm(nthreads, "diff")
    deflect_configs = {"pbdem_dem":(glm_pb, clm_dem),
                       "pbdem_zero": (glm_pb, np.zeros(np.size(glm_pb))),
                       "npbdem_dem": (-glm_pb, clm_dem),
                       "diff_zero": (glm_diff, np.zeros(np.size(glm_diff))),
                       "zero_dem":(np.zeros(np.size(glm_diff)), clm_dem)
                       }
    unl_cmb_spectra = get_unlensed_cmb_ps()
    for sim in range(nsims):
        unl_alms = get_unlensed_alms(unl_cmb_spectra, sim, unl_loc)
        _save_unl_cmbs(loc, unl_alms, sim, nthreads)
        for deflect_typ in deflect_configs:
            glm, clm = deflect_configs[deflect_typ]
            dlm = np.array([glm, clm])
            outdir = f"{loc}{sep}{deflect_typ}"

            len_maps = get_lensed_maps(dlm, unl_alms, nthreads)
            save_lens_maps(outdir, len_maps, sim)

    _save_kappa_diff(loc, glm_diff, nthreads)
    _save_omega_diff(loc, clm_diff, nthreads)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3 and len(args) != 4:
        raise ValueError(
            "Arguments should be nsims nthreads loc unl_loc")
    nsims = int(args[0])
    nthreads = int(args[1])
    loc = str(args[2])
    unl_loc = str(args[3]) if len(args)==4 else None
    main(nsims, nthreads, loc, unl_loc)
