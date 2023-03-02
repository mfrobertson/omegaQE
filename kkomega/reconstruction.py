import numpy as np
import lensit as li
from noise import Noise
from scipy.interpolate import InterpolatedUnivariateSpline
import os


class Reconstruction:

    def __init__(self, exp="SO", LDres=12, HDres=12, nsims=1, Lcuts=(30,3000,30,5000), resp_cls=None):
        if not 'LENSIT' in os.environ.keys():
            os.environ['LENSIT'] = '_tmp'
        self.noise = Noise()
        self.exp = exp
        self.LDres = LDres
        self.N_pix = 2**self.LDres
        self.resp_cls = resp_cls
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

    def Tmap(self, include_noise=True, sim=0, phi_idx=None):
        if include_noise:
            return self.maps.get_sim_tmap(sim, phi_idx) - self.maps.get_noise_sim_tmap(sim) + self.noiseMap("TT", sim)
        return self.maps.get_sim_tmap(sim, phi_idx) - self.maps.get_noise_sim_tmap(sim)

    def QUmap(self, include_noise=True, sim=0, phi_idx=None):
        if include_noise:
            return self.maps.get_sim_qumap(sim, phi_idx)[0] - self.maps.get_noise_sim_qmap(sim) + self.noiseMap("EE", sim), self.maps.get_sim_qumap(sim, phi_idx)[1] - self.maps.get_noise_sim_umap(sim) + self.noiseMap("BB", sim)
        return self.maps.get_sim_qumap(sim, phi_idx)[0] - self.maps.get_noise_sim_qmap(sim), self.maps.get_sim_qumap(sim, phi_idx)[1] - self.maps.get_noise_sim_umap(sim)

    def _get_seed(self, typ, sim):
        seed = 3 * sim
        if typ == "EE":
            seed += 1
        elif typ == "BB":
            seed += 2
        return seed

    def _get_gauss_matrix(self, shape, typ, sim):
        seed = self._get_seed(typ, sim)
        np.random.seed(seed)
        mean = 0
        var = 1 / np.sqrt(2)
        real = np.random.normal(mean, var, shape)
        imag = np.random.normal(mean, var, shape)
        return real + (1j * imag)

    def _enforce_symmetries(self, fft_map):
        # Setting divergent point to 0 (this point represents mean of real field so this is reasonable)
        fft_map[0, 0] = 0

        # Ensuring Nyquist points are real
        fft_map[self.N_pix // 2, 0] = np.real(fft_map[self.N_pix // 2, 0]) * np.sqrt(2)
        fft_map[0, self.N_pix // 2] = np.real(fft_map[0, self.N_pix // 2]) * np.sqrt(2)
        fft_map[self.N_pix // 2, self.N_pix // 2] = np.real(fft_map[self.N_pix // 2, self.N_pix // 2]) * np.sqrt(2)

        # +ve k_y mirrors -ve conj(k_y) at k_x = 0
        fft_map[self.N_pix // 2 + 1:, 0] = np.conjugate(fft_map[1:self.N_pix // 2, 0][::-1])

        # +ve k_y mirrors -ve conj(k_y) at k_x = N/2 (Nyquist freq)
        fft_map[self.N_pix // 2 + 1:, -1] = np.conjugate(fft_map[1:self.N_pix // 2, -1][::-1])
        return fft_map

    def noiseMap(self, typ, sim):
        # TODO: This return different noise map every call
        Lmax_data = 5000
        n = np.sqrt(self.noise.get_cmb_gaussian_N(typ, None, None, Lmax_data, exp=self.exp))
        Ls = np.arange(np.size(n))
        n_spline = InterpolatedUnivariateSpline(Ls, n)
        n_rfft = n_spline(self.maps.lib_datalm.ell_mat()[:2**self.LDres, :2**self.LDres//2+1])
        physical_length = np.sqrt(np.prod(self.isocov.lib_skyalm.lsides))
        gauss_matrix = self._get_gauss_matrix((2**self.LDres, 2**self.LDres//2+1), typ, sim)
        Tcmb = 2.7255
        return np.fft.irfft2(self._enforce_symmetries(n_rfft * gauss_matrix), norm="forward") * Tcmb * 1e6 / physical_length

    def _get_iblm(self, fields, include_noise, sim, phi_idx, return_data_alms=False):
        if fields == "T":
            estimator = "T"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim, phi_idx))
            iblm = self.isocov.get_iblms(estimator, np.atleast_2d(T_alm), use_cls_len=True)[0]
            if return_data_alms:
                return estimator, iblm, T_alm
            return estimator, iblm
        if fields == "EB":
            estimator = "QU"
            Qmap, Umap = self.QUmap(include_noise, sim, phi_idx)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            data_alms = np.array([Q_alm, U_alm])
            iblm = self.isocov.get_iblms(estimator, data_alms, use_cls_len=True)[0]
            if return_data_alms:
                return estimator, iblm, data_alms
            return estimator, iblm
        if fields == "TEB":
            estimator = "TQU"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise, sim, phi_idx))
            Qmap, Umap = self.QUmap(include_noise, sim, phi_idx)
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

        N0s = self.isocov.get_N0cls('QU', self.isocov.lib_skyalm, use_cls_len=False)
        H0s = [self._inverse_nonzero(N0s[0]), self._inverse_nonzero(N0s[1])]

        cls_unl = li.get_fidcls(LMAX_SKY, wrotationCls=True)[0]
        cpp_priors = [np.copy(cls_unl['pp'][:LMAX_SKY + 1]), np.copy(cls_unl['oo'][:LMAX_SKY + 1])]

        lib_skyalm = ell_mat.ffs_alm_pyFFTW(self.isocov.lib_datalm.ell_mat, filt_func=lambda ell: ell <= LMAX_SKY)

        nT, nP, beam = self._get_exp_noise(self.exp)
        nT = 1
        nP = np.sqrt(2)*nT
        beam = 0.5

        transf = gauss_beam(beam / 180. / 60. * np.pi, lmax=LMAX_SKY)  #: fiducial beam

        filt = ffs_ninv_filt_ideal.ffs_ninv_filt(self.isocov.lib_datalm, lib_skyalm, cls_unl, transf, nT, nP)

        chain_descr = chain_samples.get_isomgchain(filt.lib_skyalm.ellmax, filt.lib_datalm.shape, tol=1e-6, iter_max=200)

        opfilt = li.qcinv.opfilt_cinv_noBB
        opfilt._type = estimator

        iterator = ffs_iterator_cstMF(os.environ['LENSIT'], estimator, filt, datalms, self.isocov.lib_skyalm, POlm0, H0s, POlm0 * 0., cpp_priors, chain_descr=chain_descr, opfilt=opfilt, verbose=True)
        return iterator

    def _QE_iter(self, fields, idx, include_noise, sim, phi_idx):
        estimator, phi_alm_0, omega_alm_0, data_alms = self._iter_starting_point(fields, include_noise, sim, phi_idx)
        iter_lib = self._get_iter_instance(estimator, np.array([phi_alm_0, omega_alm_0]), data_alms)
        iter_lib.soltn_cond = True
        N_iters = 3
        for i in range(N_iters + 1):
            iter_lib.iterate(i, 'p')
        size = iter_lib.lib_qlm.alm_size
        start, end = idx*size, (idx+1)*size
        return iter_lib.get_POlm(N_iters)[start:end]

    def _QE(self, fields, idx, include_noise, sim, phi_idx):
        estimator, iblm = self._get_iblm(fields, include_noise, sim, phi_idx)
        alm_no_norm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True, resp_cls=self.resp_cls)[idx]
        f = self.isocov.get_response(estimator, self.isocov.lib_skyalm, cls_weights=self.resp_cls, cls_cmb=self.resp_cls)[idx]
        f_inv = self._inverse_nonzero(f)
        alm_norm = self.isocov.lib_skyalm.almxfl(alm_no_norm, f_inv)
        return alm_norm

    def get_phi_rec(self, fields, return_map=False, include_noise=True, sim=0, phi_idx=None, iter_rec=False):
        qe_func = self._QE_iter if iter_rec else self._QE
        self.phi = qe_func(fields, 0, include_noise, sim, phi_idx)
        if return_map:
            return self._get_rfft_map(self.phi)
        return self.phi

    def get_curl_rec(self, fields, return_map=False, include_noise=True, sim=0, phi_idx=None, iter_rec=False):
        qe_func = self._QE_iter if iter_rec else self._QE
        self.curl = qe_func(fields, 1, include_noise, sim, phi_idx)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rec = Reconstruction("SO")
    phi_rec, phi_N0 = rec.get_phi_rec("T")
    Cl_phi = Cl = rec.isocov.lib_skyalm.alm2cl(phi_rec, phi_rec, ellmax=3000)
    ell = np.where(Cl != 0.)[0]
    Cl = Cl[ell]
    plt.semilogy(ell, ell**4*Cl*1e7/(2*np.pi))
    plt.show()
