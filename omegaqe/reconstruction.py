import numpy as np
import lensit as li
from omegaqe.noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import signal
import os
import shutil
import warnings
import multiprocessing
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

class Reconstruction:

    def __init__(self, fields, exp="SO", LDres=12, HDres=12, nsims=1, Lcuts=(30,3000,30,5000), resp_cls=None):
        if not 'LENSIT' in os.environ.keys():
            os.environ['LENSIT'] = '_tmp'
        self.fields = fields
        self.noise = Noise()
        self.exp = exp
        self.LDres = LDres
        self.N_pix = 2**self.LDres
        self.Lcuts = Lcuts
        self.resp_cls = resp_cls
        exp_conf = tuple(self._get_lensit_config(self.exp, self.Lcuts))
        # n_threads = multiprocessing.cpu_count()
        n_threads = 1
        print(f"Lensit will use {n_threads} threads.")
        self.maps = li.get_maps_lib(exp_conf, LDres, HDres=HDres, cache_lenalms=False, cache_maps=False, nsims=nsims, num_threads=n_threads)
        self.isocov = li.get_isocov(exp_conf, LDres, HD_res=HDres, pyFFTWthreads=n_threads)
        self.curl = None
        self.phi = None
        self.phi_iter_norm = None
        self.curl_iter_norm = None
        self._ell_check()

    def _ell_check(self):
        ellmax_lim = 5000
        lensit_lmax = self.maps.lib_datalm.ell_mat()[2**self.LDres//2][2**self.LDres//2]
        if lensit_lmax > ellmax_lim:
            warnings.warn(f"LDres and HDres setting produce ells > {ellmax_lim}. (lensit maps have ells up to {lensit_lmax})")

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

    def _get_exp_noise(self, exp, Lmax=None):
        lensit_ellmax_sky = 6000 if Lmax is None else Lmax
        nT, beam = self.noise.get_noise_args(exp)
        if nT is not None and beam is not None:
            nT = np.ones(lensit_ellmax_sky + 1) * nT
            return nT, np.sqrt(2) * nT, beam
        return self._deconstruct_noise_curve("TT", exp, 0), self._deconstruct_noise_curve("EE", exp, 0), 0

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

    def Tmap(self, include_noise=True, sim=0, phi_idx=None, gauss=False):
        if gauss:
            return self.fields.get_cmb_map("TT", include_noise, fft=False, muK=True, sim=sim)/(2*np.pi)
        map = self.maps.get_sim_tmap(sim, phi_idx) - self.maps.get_noise_sim_tmap(sim)
        if include_noise:
            return map + self.fields.get_noise_map("TT", True, fft=False, muK=True, sim=sim)/(2*np.pi)
        return map

    def QUmap(self, include_noise=True, sim=0, phi_idx=None, gauss=False):
        if gauss:
            Qmap, Umap = self.fields.get_QU_map(include_noise, fft=False, muK=True)
            return Qmap/(2*np.pi), Umap/(2*np.pi)
        Qmap = self.maps.get_sim_qumap(sim, phi_idx)[0] - self.maps.get_noise_sim_qmap(sim)
        Umap = self.maps.get_sim_qumap(sim, phi_idx)[1] - self.maps.get_noise_sim_umap(sim)
        if include_noise:
            Enoise_fft = self.fields.get_noise_map("EE", True, fft=True, muK=True, sim=sim)
            Bnoise_fft = self.fields.get_noise_map("BB", True, fft=True, muK=True, sim=sim)
            Qnoise_fft, Unoise_fft = self.fields.EB_to_QU(Enoise_fft, Bnoise_fft)
            Qnoise = np.fft.irfft2(Qnoise_fft, norm="forward")
            Unoise = np.fft.irfft2(Unoise_fft, norm="forward")
            return Qmap + Qnoise/(2*np.pi), Umap + Unoise/(2*np.pi)
        return Qmap, Umap

    def _get_iblm(self, fields, include_noise, sim, phi_idx, return_data_alms=False, gaussCMB=False):
        if fields == "T":
            estimator = "T"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim, phi_idx, gaussCMB))
            iblm = self.isocov.get_iblms(estimator, np.atleast_2d(T_alm), use_cls_len=True)[0]
            if return_data_alms:
                return estimator, iblm, T_alm
            return estimator, iblm
        if fields == "EB":
            estimator = "QU"
            Qmap, Umap = self.QUmap(include_noise, sim, phi_idx, gaussCMB)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            data_alms = np.array([Q_alm, U_alm])
            iblm = self.isocov.get_iblms(estimator, data_alms, use_cls_len=True)[0]
            if return_data_alms:
                return estimator, iblm, data_alms
            return estimator, iblm
        if fields == "TEB":
            estimator = "TQU"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim, phi_idx, gaussCMB))
            Qmap, Umap = self.QUmap(include_noise, sim, phi_idx, gaussCMB)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            data_alms = np.array([T_alm, Q_alm, U_alm])
            iblm = self.isocov.get_iblms(estimator, data_alms, use_cls_len=True)[0]
            if return_data_alms:
                return estimator, iblm, data_alms
            return estimator, iblm
        raise ValueError(f"Supplied fields {fields} not one of T, EB, or TEB")

    def _iter_starting_point(self, fields, include_noise, sim, phi_idx):
        estimator, iblm, data_alms = self._get_iblm(fields, include_noise, sim, phi_idx, True)
        alm_no_norm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True)
        N0 = self.isocov.get_N0cls(estimator, self.isocov.lib_skyalm, use_cls_len=True)
        Cl_phi = li.get_fidcls(wrotationCls=True)[0]['pp'][:self.isocov.lib_skyalm.ellmax + 1]
        Cl_omega = li.get_fidcls(wrotationCls=True)[0]['oo'][:self.isocov.lib_skyalm.ellmax + 1]
        wiener_filter_phi = self._inverse_nonzero(self._inverse_nonzero(N0[0]) + self._inverse_nonzero(Cl_phi))
        wiener_filter_omega = self._inverse_nonzero(self._inverse_nonzero(N0[1]) + self._inverse_nonzero(Cl_omega))
        alm_norm_phi = self.isocov.lib_skyalm.almxfl(alm_no_norm[0], wiener_filter_phi)
        alm_norm_omega = self.isocov.lib_skyalm.almxfl(alm_no_norm[1], wiener_filter_omega)
        return estimator, alm_norm_phi, alm_norm_omega, data_alms

    def _get_iter_instance(self, estimator, POlm0, datalms):
        from lensit.ffs_iterators.ffs_iterator_wcurl import ffs_iterator_cstMF
        from lensit.misc.misc_utils import gauss_beam
        from lensit.qcinv import ffs_ninv_filt_ideal, chain_samples
        from lensit.ffs_covs import ell_mat
        from lensit import LMAX_SKY

        N0s = self.isocov.get_N0cls(estimator, self.isocov.lib_skyalm, use_cls_len=False)
        H0s = [self._inverse_nonzero(N0s[0]), self._inverse_nonzero(N0s[1])]

        cls_unl = li.get_fidcls(LMAX_SKY, wrotationCls=True)[0]
        cpp_priors = [np.copy(cls_unl['pp'][:LMAX_SKY + 1]), np.copy(cls_unl['oo'][:LMAX_SKY + 1])]

        lib_skyalm = ell_mat.ffs_alm_pyFFTW(self.isocov.lib_datalm.ell_mat, filt_func=lambda ell: ell <= LMAX_SKY)

        nT, nP, beam = self._get_exp_noise(self.exp)
        # _,_, nT, nP = self._apply_Lcuts(self.Lcuts, nT, nP)    #TODO: Should the Lcuts be applied? (doesn't work)

        transf = gauss_beam(beam / 180. / 60. * np.pi, lmax=LMAX_SKY)  #: fiducial beam

        filt = ffs_ninv_filt_ideal.ffs_ninv_filt(self.isocov.lib_datalm, lib_skyalm, cls_unl, transf, nT, nP)

        chain_descr = chain_samples.get_isomgchain(filt.lib_skyalm.ellmax, filt.lib_datalm.shape, tol=1e-6, iter_max=200)

        opfilt = li.qcinv.opfilt_cinv_noBB
        opfilt._type = estimator
        iter_dir = os.path.join(os.environ['LENSIT'],"iter")
        if os.path.exists(iter_dir) and os.path.isdir(iter_dir):
            shutil.rmtree(iter_dir)
        iterator = ffs_iterator_cstMF(iter_dir, estimator, filt, datalms, self.isocov.lib_skyalm, POlm0, H0s, POlm0 * 0., cpp_priors, chain_descr=chain_descr, opfilt=opfilt, verbose=True)
        return iterator

    def _gaussian_smooth_1d(self, data, sigma, nsigma=5):
        win = signal.gaussian(nsigma * sigma, std=sigma)
        res = np.convolve(data, win, mode='same')
        return res / np.convolve(np.ones(data.shape), win, mode='same')

    def _get_iter_normalisation(self, alm, idx, sim, phi_idx):
        phi_idx = sim if phi_idx is None else phi_idx
        alm_input = self.maps.lencmbs.get_sim_plm(phi_idx) if idx==0 else self.maps.lencmbs.get_sim_olm(sim)
        alm_input = self.isocov.lib_skyalm.udgrade(self.maps.lencmbs.lib_skyalm, alm_input)
        cl_cross = self.isocov.lib_skyalm.alm2cl(alm, alm_input)
        cl_corr = self._inverse_nonzero(cl_cross) * self.isocov.lib_skyalm.alm2cl(alm_input)
        Ls = np.where(cl_corr != 0)[0]
        cl_corr = self._gaussian_smooth_1d(cl_corr[Ls], 50)
        return InterpolatedUnivariateSpline(Ls, cl_corr)

    def _renorm(self, alm, idx, sim, phi_idx):
        #TODO: finish this...
        return alm

    def _QE_iter(self, fields, idx, include_noise, sim, phi_idx, renorm=False):
        estimator, phi_alm_0, omega_alm_0, data_alms = self._iter_starting_point(fields, include_noise, sim, phi_idx)
        iter_lib = self._get_iter_instance(estimator, np.array([phi_alm_0, omega_alm_0]), data_alms)
        iter_lib.soltn_cond = True
        N_iters = 3
        for i in range(N_iters + 1):
            iter_lib.iterate(i)
        size = iter_lib.lib_qlm.alm_size
        start, end = idx*size, (idx+1)*size
        alm = iter_lib.get_POlm(N_iters)[start:end]
        if renorm:
            return self._renorm(alm, idx, sim, phi_idx)
        return alm

    def _QE(self, fields, idx, include_noise, sim, phi_idx, gaussCMB, diffSims, diffSim_offset):
        print(f"Quadratic estimation of type: {fields}, with noise: {include_noise}, phi_idx: {phi_idx}, gauss fields: {gaussCMB}, N1: {diffSims}, diff_offset: {diffSim_offset}")
        if diffSims:
            phi_idx = sim if phi_idx is None else phi_idx
            estimator, iblm = self._get_iblm(fields, include_noise, sim, phi_idx, gaussCMB=gaussCMB)
            estimator, iblm2 = self._get_iblm(fields, include_noise, sim + diffSim_offset, phi_idx, gaussCMB=gaussCMB)
            alm_no_norm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True, resp_cls=self.resp_cls, iblms2=iblm2)[idx]
            alm_no_norm += 0.5 * self.isocov.get_qlms(estimator, iblm2, self.isocov.lib_skyalm, use_cls_len=True, resp_cls=self.resp_cls, iblms2=iblm)[idx]
        else:
            estimator, iblm = self._get_iblm(fields, include_noise, sim, phi_idx, gaussCMB=gaussCMB)
            alm_no_norm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True, resp_cls=self.resp_cls)[idx]
        f = self.isocov.get_response(estimator, self.isocov.lib_skyalm, cls_weights=self.resp_cls, cls_cmb=self.resp_cls)[idx]
        f_inv = self._inverse_nonzero(f)
        alm_norm = self.isocov.lib_skyalm.almxfl(alm_no_norm, f_inv)
        return alm_norm

    def get_phi_rec(self, fields, return_map=False, include_noise=True, sim=0, phi_idx=None, iter_rec=False, gaussCMB=False, diffSims=True, diffSim_offset=1):
        print(f"Performing Lensit phi reconstruction on sim {sim}...")
        # qe_func = self._QE_iter if iter_rec else self._QE
        qe_func = self._QE
        self.phi = qe_func(fields, 0, include_noise, sim, phi_idx, gaussCMB, diffSims, diffSim_offset)
        if return_map:
            return self._get_rfft_map(self.phi)
        return self.phi

    def get_curl_rec(self, fields, return_map=False, include_noise=True, sim=0, phi_idx=None, iter_rec=False, gaussCMB=False, diffSims=True, diffSim_offset=1):
        print(f"Performing Lensit curl reconstruction on sim {sim}...")
        # qe_func = self._QE_iter if iter_rec else self._QE
        qe_func = self._QE
        self.curl = qe_func(fields, 1, include_noise, sim, phi_idx, gaussCMB, diffSims, diffSim_offset)
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
        physical_length = np.sqrt(np.prod(self.isocov.lib_skyalm.lsides))  # Why is this needed?
        return rfft * physical_length
