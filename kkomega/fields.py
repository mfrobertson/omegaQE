import numpy as np
from fisher import Fisher
from template import Template
import postborn
from reconstruction import Reconstruction
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
import copy


class Fields:

    def __init__(self, fields, exp, N_pix_pow=10, kmax=5000, setup_cmb_lens_rec=False, HDres=None, Nsims=1, sim=0, resp_cls=None):
        # TODO: Lensit has ellM = int(np.around(kM - 1/2))?? So my maps disagree with lensit of small scales...
        self.N_pix = 2**N_pix_pow
        self.HDres = HDres
        self.kmax_map = self._get_kmax(kmax)                # If HDres is not None then HDres determines kmax_map
        self.kmax_map_round = int(np.floor(self.kmax_map))
        self.kM, self.k_values = self._get_k_values()
        self.fish = Fisher()
        self.covariance = self.fish.covariance
        self.covariance.setup_cmb_noise(exp, "TEB", True, "gradient", 30, 3000, 30, 5000, False, False, data_dir="data")
        self.rec = None
        input_kappa_map = None
        enforce_sym = True
        if setup_cmb_lens_rec:
            self.rec = Reconstruction(exp, LDres=N_pix_pow, HDres=HDres, nsims=Nsims, resp_cls=resp_cls)
            self._sim = sim
            input_kappa_map = self._get_lensit_kappa_map(sim=self._sim)
            enforce_sym = False
        self.fields = self._get_rearanged_fields(fields) if ("k" in fields and setup_cmb_lens_rec) else self._get_fields(fields)
        self.template = None
        self.y = self._get_y(input_kappa_map)
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True, enforce_sym=enforce_sym)
            self.fft_noise_maps[field] = self.get_noise_map(field)

    def change_sim(self, sim):
        self._sim = sim
        input_kappa_map = self._get_lensit_kappa_map(sim=self._sim)
        enforce_sym = False
        self.y = self._get_y(input_kappa_map)
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True, enforce_sym=enforce_sym)
            self.fft_noise_maps[field] = self.get_noise_map(field)

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, iter_ext=None, data_dir=None):
        return self.fish.setup_noise(exp, qe, gmv, ps, L_cuts, iter, iter_ext, data_dir)

    def _get_lensit_kappa_map(self, sim=0):
        phi_map = 2 * np.pi * self.rec.get_phi_input(return_map=True, sim=sim)
        return phi_map * self.kM ** 2 / 2

    def _get_lensit_dist(self, HDres):
        return np.sqrt(4.*np.pi)/(2**14) * (2**(int(HDres-np.log2(self.N_pix)))) * self.N_pix

    def _get_kmax(self, kmax):
        if self.HDres is None:
            return kmax
        kmax = np.sqrt(2) * self.N_pix * np.pi / self._get_lensit_dist(self.HDres)
        return kmax     # Lensit do this ell = |k|-1/2, don't know why???

    def _get_fields(self, fields):
        return np.char.array(list(fields))

    def _get_rearanged_fields(self, fields):
        fields = self._get_fields(fields)
        fields = fields[fields != "k"]
        fields = np.insert(fields, 0, "k")
        return fields

    def _get_cov(self):
        N_fields = np.size(self.fields)
        C = np.empty((self.kmax_map_round, N_fields, N_fields))
        for iii, field_i in enumerate(self.fields):
            for jjj, field_j in enumerate(self.fields):
                C[:, iii, jjj] = self.covariance.get_Cl(field_i + field_j, ellmax=self.kmax_map_round)[1:]
        return C * (2*np.pi)**2

    def _get_L(self, C):
        N_fields = np.size(self.fields)
        ks_sample = np.arange(1, self.kmax_map_round + 1)
        L = np.linalg.cholesky(C)
        L_new = np.empty((np.size(self.k_values), N_fields, N_fields))
        for iii in range(N_fields):
            for jjj in range(N_fields):
                L_ij = L[:, iii, jjj]
                L_new[:, iii, jjj] = InterpolatedUnivariateSpline(ks_sample, L_ij)(self.k_values)
        return L_new

    def _get_gauss_matrix(self, shape):
        mean = 0
        var = 1 / np.sqrt(2)
        real = np.random.normal(mean, var, shape)
        imag = np.random.normal(mean, var, shape)
        return real + (1j * imag)

    def _get_y(self, kappa_map=None):
        C = self._get_cov()
        N_fields = np.size(self.fields)
        L = self._get_L(C)
        v = self._get_gauss_matrix((np.size(self.k_values), N_fields, 1))
        if kappa_map is not None:
            C_kappa_sqrt = L[:, 0, 0]
            v[:, 0, 0] = kappa_map.flatten() / C_kappa_sqrt
        y = np.matmul(L, v)
        return y

    def get_dist(self):
        return np.sqrt(2) * self.N_pix * np.pi / self.kmax_map

    def get_kx_ky(self, N_pix=None, dist=None):
        if N_pix is None:
            N_pix = self.N_pix
        if dist is None:
            dist = self.get_dist()
        sep = dist/N_pix
        kx = np.fft.rfftfreq(N_pix, sep) * 2 * np.pi
        ky = np.fft.fftfreq(N_pix, sep) * 2 * np.pi
        return kx, ky

    def get_k_matrix(self, Npix=None, dist=None):
        kx, ky = self.get_kx_ky(Npix, dist)
        kSqr = kx[np.newaxis, ...] ** 2 + ky[..., np.newaxis] ** 2
        # return np.rint(np.sqrt(kSqr) - 0.5)     # This is what lensit does
        return np.sqrt(kSqr)

    def _get_k_values(self):
        kM = self.get_k_matrix()
        k_values = kM.flatten()
        return kM, k_values

    def _get_index(self, field):
        return np.where(self.fields == field)[0][0]

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

    def get_map(self, field, fft=True, enforce_sym=True):
        index = self._get_index(field)
        y = self.y
        fft_map = copy.deepcopy(y[:, index, 0])
        fft_map = np.reshape(fft_map, (np.shape(self.kM)))

        if enforce_sym:
            fft_map = self._enforce_symmetries(fft_map)

        if not fft:
            return np.fft.irfft2(fft_map, norm="forward")     # Should check the normalisation
        return fft_map

    def _get_N(self, field):
        kmax = 5000
        if field == "k":
            return self.covariance.noise.get_N0("kappa", ellmax=kmax, recalc_N0=False)
        if field == "g":
            return self.covariance.noise.get_gal_shot_N(ellmax=kmax)
        if field == "I":
            N_dust = self.covariance.noise.get_dust_N(353e9, ellmax=kmax)
            N_cib = self.covariance.noise.get_cib_shot_N(353e9, ellmax=kmax)
            N = N_dust+N_cib
            return N

    def get_noise_map(self, field):
        N = self._get_N(field)
        Ls = np.arange(np.size(N))
        N_spline = InterpolatedUnivariateSpline(Ls[2:], N[2:])
        gauss_matrix = self._get_gauss_matrix(np.shape(self.kM))
        return self._enforce_symmetries(np.sqrt(N_spline(self.kM) * (2*np.pi)**2) * gauss_matrix)

    def get_ps(self, rfft_map1, rfft_map2=None, kmin=1, kmax=None, kM=None):
        kM = self.kM if kM is None else kM
        if kmax is None:
            kmax = int(np.floor(self.kmax_map/np.sqrt(2)))
        N_pix = np.shape(kM)[0]
        fft_map1 = copy.deepcopy(rfft_map1)
        if rfft_map2 is None:
            fft_map2 = copy.deepcopy(fft_map1)
        else:
            fft_map2 = copy.deepcopy(rfft_map2)

        ps_raw = np.real(np.conjugate(fft_map1) * fft_map2)
        k_counts = np.bincount(kM[:, 1:-1].flatten().astype(int))
        k_counts += np.bincount(kM[1:N_pix // 2, [0, -1]].flatten().astype(int))
        ps = np.bincount(kM[:, 1:-1].flatten().astype(int), weights=ps_raw[:, 1:-1].flatten())
        ps += np.bincount(kM[1:N_pix // 2, [0, -1]].flatten().astype(int), weights=ps_raw[1:N_pix // 2, [0, -1]].flatten())

        ps = ps / k_counts
        ps[k_counts == 0] = 0
        ps = ps[:kmax + 1]
        ps = ps[kmin:]
        ks = np.arange(kmin, kmax+1)

        return ks, ps

    def get_ps_binned(self, rfft_map1, rfft_map2=None, nBins=20, kmin=1, kmax=None, kM=None):
        ks, ps = self.get_ps(rfft_map1, rfft_map2, kmin, kmax, kM)
        means, bin_edges, binnumber = stats.binned_statistic(ks, ps, 'mean', bins=nBins)
        binSeperation = bin_edges[1] - bin_edges[0]
        kBins = np.asarray([bin_edges[i] - binSeperation / 2 for i in range(1, len(bin_edges))])
        counts, *others = stats.binned_statistic(ks, ps, 'count', bins=nBins)
        stds, *others = stats.binned_statistic(ks, ps, 'std', bins=nBins)
        errors = stds / np.sqrt(counts)
        return means, kBins, errors

    def get_omega_rec(self, cmb_fields="T", include_noise=True, phi_idx=None, iter_rec=False):
        if self.rec is None:
            raise ValueError(f"CMB lensing reconstruction not setup for this Fields instance.")
        curl_map = 2 * np.pi * self.rec.get_curl_rec(cmb_fields, return_map=True, include_noise=include_noise, sim=self._sim, phi_idx=phi_idx, iter_rec=iter_rec)
        return curl_map * self.kM **2 / 2

    def get_kappa_rec(self, cmb_fields="T", include_noise=True, phi_idx=None, iter_rec=False):
        if self.rec is None:
            raise ValueError(f"CMB lensing reconstruction not setup for this Fields instance.")
        kappa_map = 2 * np.pi * self.rec.get_phi_rec(cmb_fields, return_map=True, include_noise=include_noise, sim=self._sim, phi_idx=phi_idx, iter_rec=iter_rec)
        return kappa_map * self.kM **2 / 2

    def get_omega_fiducial(self):
        omega_Ls = self.fish.covariance.get_log_sample_Ls(2, self.kmax_map_round, 100, dL_small=2)
        C_omega = postborn.omega_ps(omega_Ls)
        C_omega_spline = InterpolatedUnivariateSpline(omega_Ls, C_omega)
        gauss_matrix = self._get_gauss_matrix(np.shape(self.kM))
        return self._enforce_symmetries(np.sqrt(C_omega_spline(self.kM) * (2 * np.pi) ** 2) * gauss_matrix)

    def get_omega_template(self, Nchi=20, F_L_spline=None, C_inv_spline=None, tracer_noise=False, reinitialise=False, use_kappa_rec=False):
        if self.template is None or reinitialise:
            self.template = Template(self, Lmin=30, Lmax=3000, F_L_spline=F_L_spline, C_inv_spline=C_inv_spline, tracer_noise=tracer_noise, use_kappa_rec=use_kappa_rec)
        return self.template.get_omega(Nchi)

