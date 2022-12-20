import numpy as np
import lensit as li
import os


class Reconstruction:

    def __init__(self, exp="SO", LDres=12, HDres=12):
        if not 'LENSIT' in os.environ.keys():
            os.environ['LENSIT'] = '_tmp'
        self.exp = exp
        exp_conf = tuple(self._get_lensit_config(exp))
        self.maps = li.get_maps_lib(exp_conf, LDres, HDres=HDres, cache_lenalms=False, cache_maps=False, nsims=1, num_threads=4)
        self.isocov = li.get_isocov(exp_conf, LDres, HD_res=HDres, pyFFTWthreads=4)
        self.P_alm = self.isocov.lib_skyalm.udgrade(self.maps.lencmbs.lib_skyalm, self.maps.lencmbs.get_sim_plm(0))
        self.curl = None
        self.phi = None

    def _get_exp_noise(self, exp):
        if exp == "SO":
            return 3, 3
        elif exp == "S4":
            return 1, 3
        elif exp == "HD":
            return 0.5, 0.25
        else:
            raise ValueError(f"Experiment {exp} unexpected.")

    def _get_lensit_config(self, exp):
        delta_T, beam = self._get_exp_noise(exp)
        Lmin = 30
        Lmax = 3000
        return "custom", delta_T, beam, Lmin, Lmax

    def _inverse_nonzero(self, array):
        ret = np.zeros_like(array)
        ret[np.where(array != 0.)] = 1. / array[np.where(array != 0.)]
        return ret

    def Tmap(self, include_noise=True):
        if include_noise:
            return self.maps.get_sim_tmap(0)
        return self.maps.get_sim_tmap(0) - self.maps.get_noise_sim_tmap(0)

    def QUmap(self, include_noise=True):
        if include_noise:
            return self.maps.get_sim_qumap(0)
        return self.maps.get_sim_qumap(0)[0] - self.maps.get_noise_sim_qmap(0), self.maps.get_sim_qumap(0)[1] - self.maps.get_noise_sim_umap(0)

    def _get_iblm(self, fields, include_noise):
        if fields == "T":
            estimator = "T"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise))
            iblm = self.isocov.get_iblms(estimator, np.atleast_2d(T_alm), use_cls_len=True)[0]
            return estimator, iblm
        if fields == "EB":
            estimator = "QU"
            Qmap, Umap = self.QUmap(include_noise)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            iblm = self.isocov.get_iblms(estimator, np.array([Q_alm, U_alm]), use_cls_len=True)[0]
            return estimator, iblm
        if fields == "TEB":
            estimator = "TQU"
            T_alm = self.isocov.lib_datalm.map2alm(self.Tmap(include_noise))
            Qmap, Umap = self.QUmap(include_noise)
            Q_alm = self.isocov.lib_datalm.map2alm(Qmap)
            U_alm = self.isocov.lib_datalm.map2alm(Umap)
            iblm = self.isocov.get_iblms(estimator, np.array([T_alm, Q_alm, U_alm]), use_cls_len=True)[0]
            return estimator, iblm
        raise ValueError(f"Supplied fields {fields} not one of T, EB, or TEB")

    def _QE(self, fields, idx, include_noise):
        estimator, iblm = self._get_iblm(fields, include_noise)
        alm = 0.5 * self.isocov.get_qlms(estimator, iblm, self.isocov.lib_skyalm, use_cls_len=True)[idx]
        N0 = self.isocov.get_N0cls(estimator, self.isocov.lib_skyalm)[idx]
        f = self.isocov.get_response(estimator, self.isocov.lib_skyalm)[idx]
        f_inv = self._inverse_nonzero(f)
        phi = self.isocov.lib_skyalm.almxfl(alm, f_inv)
        return phi, N0

    def get_phi_rec(self, fields, return_map=False, include_noise=True):
        self.phi, N0 = self._QE(fields, 0, include_noise)
        if return_map:
            return self.isocov.lib_skyalm.alm2map(self.phi), N0
        return self.phi, N0

    def get_curl_rec(self, fields, return_map=False, include_noise=True):
        self.curl, N0 = self._QE(fields, 1, include_noise)
        if return_map:
            return self.isocov.lib_skyalm.alm2map(self.curl), N0
        return self.curl, N0

    def get_phi_input(self, return_map=False):
        if return_map:
            return self.isocov.lib_skyalm.alm2map(self.P_alm)
        return self.P_alm


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rec = Reconstruction("SO")
    phi_rec, phi_N0 = rec.get_phi_rec("T")
    Cl_phi = Cl = rec.isocov.lib_skyalm.alm2cl(phi_rec, phi_rec, ellmax=3000)
    ell = np.where(Cl != 0.)[0]
    Cl = Cl[ell]
    plt.semilogy(ell, ell**4*Cl*1e7/(2*np.pi))
    plt.show()
