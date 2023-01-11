import numpy as np
import lensit as li
from noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
import os


class Reconstruction:

    def __init__(self, exp="SO", LDres=12, HDres=12, nsims=1, Lcuts=(30,3000,30,5000)):
        if not 'LENSIT' in os.environ.keys():
            os.environ['LENSIT'] = '_tmp'
        self.noise = Noise()
        self.exp = exp
        exp_conf = tuple(self._get_lensit_config(exp, Lcuts))
        self.maps = li.get_maps_lib(exp_conf, LDres, HDres=HDres, cache_lenalms=False, cache_maps=False, nsims=nsims, num_threads=4)
        self.isocov = li.get_isocov(exp_conf, LDres, HD_res=HDres, pyFFTWthreads=4)
        self.curl = None
        self.phi = None

    def _deconstruct_noise_curve(self, typ, exp, beam):
        Lmax_data = 5000
        N = self.noise.get_cmb_gaussian_N(typ, None, None, Lmax_data, exp)
        T_cmb = 2.7255
        arcmin_to_rad = np.pi / 180 / 60
        Ls = np.arange(np.size(N))
        beam *= arcmin_to_rad
        deconvolve_beam = np.exp(Ls*(Ls+1)*beam**2/(8*np.log(2)))
        n = np.sqrt(N/deconvolve_beam) * T_cmb / 1e-6 / arcmin_to_rad
        lensit_ellmax_sky = 6000
        Ls_sample = np.arange(np.size(N))
        Ls = np.arange(lensit_ellmax_sky + 1)
        return InterpolatedUnivariateSpline(Ls_sample, n)(Ls)

    def _get_exp_noise(self, exp):
        lensit_ellmax_sky = 6000
        if exp == "SO":
            nT = 3
            nT = np.ones(lensit_ellmax_sky + 1) * nT
            nP = np.sqrt(2) * nT
            beam = 3
            return nT, nP, beam
        if exp == "SO_base":
            beam = 0
            nT = self._deconstruct_noise_curve("TT", exp, beam)
            nP = self._deconstruct_noise_curve("EE", exp, beam)
            return nT, nP, beam
        if exp == "SO_goal":
            beam = 0
            nT = self._deconstruct_noise_curve("TT", exp, beam)
            nP = self._deconstruct_noise_curve("EE", exp, beam)
            return nT, nP, beam
        if exp == "S4":
            nT = 1
            nT = np.ones(lensit_ellmax_sky + 1) * nT
            nP = np.sqrt(2) * nT
            beam = 3
            return nT, nP, beam
        if exp == "S4_base":
            beam = 0
            nT = self._deconstruct_noise_curve("TT", exp, beam)
            nP = self._deconstruct_noise_curve("EE", exp, beam)
            return nT, nP, beam
        if exp == "HD":
            nT = 0.5
            nT = np.ones(lensit_ellmax_sky + 1) * nT
            nP = np.sqrt(2) * nT
            beam = 0.25
            return nT, nP, beam
        raise ValueError(f"Experiment {exp} unexpected.")

    def _get_Lcuts(self, T_Lmin, T_Lmax, P_Lmin, P_Lmax, strict):
        if strict:
            return np.max((T_Lmin, P_Lmin)), np.min((T_Lmax, P_Lmax))
        return np.min((T_Lmin, P_Lmin)), np.max((T_Lmax, P_Lmax))

    def _apply_Lcuts(self, Lcuts, delta_T, delta_P):
        T_Lmin = Lcuts[0]
        T_Lmax = Lcuts[1]
        P_Lmin = Lcuts[2]
        P_Lmax = Lcuts[3]
        if np.size(delta_T) == 1:
            Lmin, Lmax = self._get_Lcuts(T_Lmin, T_Lmax, P_Lmin, P_Lmax, strict=True)
            return Lmin, Lmax, delta_T, delta_P
        delta_T[:T_Lmin] = 1e10
        delta_T[T_Lmax + 1:] = 1e10
        delta_P[:P_Lmin] = 1e10
        delta_P[P_Lmax + 1:] = 1e10
        Lmin, Lmax = self._get_Lcuts(T_Lmin, T_Lmax, P_Lmin, P_Lmax, strict=False)
        return Lmin, Lmax, delta_T, delta_P


    def _get_lensit_config(self, exp, Lcuts):
        delta_T, delta_P, beam = self._get_exp_noise(exp)
        Lmin, Lmax, delta_T, delta_P = self._apply_Lcuts(Lcuts, delta_T, delta_P)
        return "custom", delta_T, beam, Lmin, Lmax, delta_P, exp

    def _inverse_nonzero(self, array):
        ret = np.zeros_like(array)
        ret[np.where(array != 0.)] = 1. / array[np.where(array != 0.)]
        return ret

    def Tmap(self, include_noise=True, sim=0):
        if include_noise:
            return self.maps.get_sim_tmap(sim)
        return self.maps.get_sim_tmap(sim) - self.maps.get_noise_sim_tmap(sim)

    def QUmap(self, include_noise=True, sim=0):
        if include_noise:
            return self.maps.get_sim_qumap(sim)
        return self.maps.get_sim_qumap(sim)[0] - self.maps.get_noise_sim_qmap(sim), self.maps.get_sim_qumap(sim)[1] - self.maps.get_noise_sim_umap(sim)

    def _get_iblm(self, fields, include_noise, sim):
        if fields == "T":
            estimator = "T"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim))
            iblm = self.isocov.get_iblms(estimator, np.atleast_2d(T_alm), use_cls_len=True)[0]
            return estimator, iblm
        if fields == "EB":
            estimator = "QU"
            Qmap, Umap = self.QUmap(include_noise, sim)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            iblm = self.isocov.get_iblms(estimator, np.array([Q_alm, U_alm]), use_cls_len=True)[0]
            return estimator, iblm
        if fields == "TEB":
            estimator = "TQU"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim))
            Qmap, Umap = self.QUmap(include_noise, sim)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            iblm = self.isocov.get_iblms(estimator, np.array([T_alm, Q_alm, U_alm]), use_cls_len=True)[0]
            return estimator, iblm
        raise ValueError(f"Supplied fields {fields} not one of T, EB, or TEB")

    def _QE(self, fields, idx, include_noise, sim):
        estimator, iblm = self._get_iblm(fields, include_noise, sim)
        alm_no_norm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True)[idx]
        f = self.isocov.get_response(estimator, self.isocov.lib_skyalm)[idx]
        f_inv = self._inverse_nonzero(f)
        alm_norm = self.isocov.lib_skyalm.almxfl(alm_no_norm, f_inv)
        return alm_norm

    def get_phi_rec(self, fields, return_map=False, include_noise=True, sim=0):
        self.phi = self._QE(fields, 0, include_noise, sim)
        if return_map:
            return self._get_rfft_map(self.phi)
        return self.phi

    def get_curl_rec(self, fields, return_map=False, include_noise=True, sim=0):
        self.curl = self._QE(fields, 1, include_noise, sim)
        if return_map:
            return self._get_rfft_map(self.curl)
        return self.curl

    def get_phi_input(self, return_map=False, sim=0):
        P_alm = self.isocov.lib_skyalm.udgrade(self.maps.lencmbs.lib_skyalm, self.maps.lencmbs.get_sim_plm(sim))
        if return_map:
            return self._get_rfft_map(P_alm)
        return P_alm

    def _get_rfft_map(self, alm):
        map = self.isocov.lib_skyalm.alm2map(alm)
        rfft = np.fft.rfft2(map, norm="forward")
        physical_length = np.sqrt(np.prod(self.isocov.lib_skyalm.lsides))
        return rfft * physical_length


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rec = Reconstruction("SO")
    phi_rec, phi_N0 = rec.get_phi_rec("T")
    Cl_phi = Cl = rec.isocov.lib_skyalm.alm2cl(phi_rec, phi_rec, ellmax=3000)
    ell = np.where(Cl != 0.)[0]
    Cl = Cl[ell]
    plt.semilogy(ell, ell**4*Cl*1e7/(2*np.pi))
    plt.show()
