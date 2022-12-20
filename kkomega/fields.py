import numpy as np
from fisher import Fisher
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
import copy


class Fields:

    def __init__(self, fields, exp="SO", N_pix=2**7, kmax=5000, kappa_map=None, HDres=None):
        if kappa_map is None:
            self.fields = self._get_fields(fields)
            self.N_pix = N_pix
            self.enforce_real = True
        else:
            self.fields = self._get_rearanged_fields(fields)
            self.N_pix = np.shape(kappa_map)[0]
            self.enforce_real = False
        self.HDres = HDres
        self.kmax_map = self._get_kmax(kmax)
        self.kM, self.k_values = self._get_k_values()
        self.fish = Fisher()
        self.covariance = self.fish.covariance
        self.covariance.setup_cmb_noise(exp, "TEB", True, "gradient", 30, 3000, 30, 5000, False, data_dir="data")
        self.y = self._get_y(kappa_map)
        self.maps = dict.fromkeys(self.fields)
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.maps[field] = self.get_map(field, fft=False)
            self.fft_maps[field] = self.get_map(field, fft=True)
            self.fft_noise_maps[field] = self.get_noise_map(field)

    def _get_kmax(self, kmax):
        if self.HDres is None:
            return kmax
        return np.sqrt(2) * self.N_pix * 180 * 60 / 0.74 / (2**self.HDres)

    def _get_fields(self, fields):
        return np.char.array(list(fields))

    def _get_rearanged_fields(self, fields):
        fields = self._get_fields(fields)
        fields = fields[fields != "k"]
        fields = np.insert(fields, 0, "k")
        return fields

    def _get_cov(self):
        N_fields = np.size(self.fields)
        C = np.empty((self.kmax_map, N_fields, N_fields))
        for iii, field_i in enumerate(self.fields):
            for jjj, field_j in enumerate(self.fields):
                C[:, iii, jjj] = self.covariance.get_Cl(field_i + field_j, ellmax=self.kmax_map)[1:]
        return C * (2*np.pi)**2

    def _get_L(self, C):
        N_fields = np.size(self.fields)
        ks_sample = np.arange(1, self.kmax_map + 1)
        L = np.linalg.cholesky(C)
        L_new = np.empty((np.size(self.k_values), N_fields, N_fields))
        for iii in range(N_fields):
            for jjj in range(N_fields):
                L_ij = L[:, iii, jjj]
                L_new[:, iii, jjj] = InterpolatedUnivariateSpline(ks_sample, L_ij)(self.k_values)
        return L_new

    def _get_y(self, kappa_map=None):
        C = self._get_cov()
        N_fields = np.size(self.fields)
        L = self._get_L(C)
        mean = 0
        var = 1 / np.sqrt(2)
        real = np.random.normal(mean, var, (np.size(self.k_values), N_fields, 1))
        imag = np.random.normal(mean, var, (np.size(self.k_values), N_fields, 1))
        v = real + (1j * imag)
        if kappa_map is not None:
            C_kappa_sqrt = L[:,0,0]
            v[:, 0, 0] = np.fft.rfft2(kappa_map, norm="forward").flatten() / C_kappa_sqrt    # Converting a real kappa map from lensit, should check what normalisation should be
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

    def get_map(self, field, fft=False):
        index = self._get_index(field)
        fft_map = copy.deepcopy(self.y[:, index, 0])
        fft_map = np.reshape(fft_map, (np.shape(self.kM)))

        if self.enforce_real:
            fft_map = self._enforce_symmetries(fft_map)

        if not fft:
            return np.fft.irfft2(fft_map, norm="forward")     # Should check the normalisation
        return fft_map

    def _get_N(self, field):
        if field == "k":
            return self.covariance.noise.get_N0("kappa", self.kmax_map, recalc_N0=False)
        if field == "g":
            return self.covariance.noise.get_gal_shot_N(ellmax=self.kmax_map)
        if field == "I":
            N_dust = self.covariance.noise.get_dust_N(353e9, ellmax=self.kmax_map)
            N_cib = self.covariance.noise.get_cib_shot_N(353e9, ellmax=self.kmax_map)
            N = N_dust+N_cib
            return N

    def get_noise_map(self, field):
        N = self._get_N(field)
        Ls = np.arange(np.size(N))
        N_spline = InterpolatedUnivariateSpline(Ls[2:], N[2:])
        mean = 0
        var = 1 / np.sqrt(2)
        real = np.random.normal(mean, var, np.shape(self.kM))
        imag = np.random.normal(mean, var, np.shape(self.kM))
        gauss_matrix = real + (1j * imag)
        return self._enforce_symmetries(np.sqrt(N_spline(self.kM) * (2*np.pi)**2) * gauss_matrix)

    def get_ps(self, rfft_map1, rfft_map2=None, kmin=1, kmax=None, kM=None):
        kM = self.kM if kM is None else kM
        if kmax is None:
            kmax = self.kmax_map/np.sqrt(2)
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

    def get_lensit_kM(self, LDres=14, HDres=14):
        N_pix = 2**LDres
        dist = 0.74 * 2**HDres * np.pi / 180 / 60
        return self.get_k_matrix(N_pix, dist)
