import numpy as np
from covariance import Covariance
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
import copy


class Fields:


    def __init__(self, fields, N_pix=2**7, kmax=5000, kappa_map=None):
        if kappa_map is None:
            self.fields = self._get_fields(fields)
            self.N_pix = N_pix
            self.enforce_real = True
        else:
            self.fields = self._get_rearanged_fields(fields)
            self.N_pix = np.shape(kappa_map)[0]
            self.enforce_real = False
        self.kmax = kmax
        self.kM, self.k_values = self._get_k_values()
        self.covariance = Covariance()
        self.y = self._get_y(kappa_map)
        self.maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.maps[field] = self.get_map(field, fft=False)
        self.fft_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True)

    def _get_fields(self, fields):
        return np.char.array(list(fields))


    def _get_rearanged_fields(self, fields):
        fields = self._get_fields(fields)
        fields = fields[fields != "k"]
        fields = np.insert(fields, 0, "k")
        return fields

    def _get_cov(self):
        N_fields = np.size(self.fields)
        C = np.empty((self.kmax, N_fields, N_fields))
        for iii, field_i in enumerate(self.fields):
            for jjj, field_j in enumerate(self.fields):
                C[:, iii, jjj] = self.covariance.get_Cl(field_i + field_j, ellmax=self.kmax)[1:]
        return C

    def _get_L(self, C):
        N_fields = np.size(self.fields)
        ks_sample = np.arange(1, self.kmax + 1)
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
            v[:, 0, 0] = np.fft.rfft2(kappa_map, norm="ortho").flatten() / C_kappa_sqrt
        y = np.matmul(L, v)
        return y

    def get_dist(self):
        return (np.sqrt(2) * self.N_pix * np.pi) / (self.kmax)

    def get_kx_ky(self, N_pix, dx):
        kx = np.fft.rfftfreq(N_pix, dx) * 2 * np.pi
        ky = np.fft.fftfreq(N_pix, dx) * 2 * np.pi
        return kx, ky

    def get_k_matrix(self, N_pix, dx):
        kx, ky = self.get_kx_ky(N_pix, dx)
        kSqr = kx[np.newaxis, ...] ** 2 + ky[..., np.newaxis] ** 2
        return np.sqrt(kSqr)

    def _get_k_values(self):
        N_pix = self.N_pix
        dist = self.get_dist()
        dx = dist / N_pix
        kM = self.get_k_matrix(N_pix, dx)
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
            return np.fft.irfft2(fft_map, norm="forward")
        return fft_map

    def get_ps(self, fields, nBins=20):
        fft_map1 = self.get_map(fields[0], fft=True)
        fft_map2 = self.get_map(fields[1], fft=True)
        ps = np.real(np.conjugate(fft_map1) * fft_map2).flatten()
        means, bin_edges, binnumber = stats.binned_statistic(self.k_values, ps, 'mean', bins=nBins)
        binSeperation = bin_edges[1]
        kBins = np.asarray([bin_edges[i] - binSeperation / 2 for i in range(1, len(bin_edges))])
        counts, *others = stats.binned_statistic(self.k_values, ps, 'count', bins=nBins)
        stds, *others = stats.binned_statistic(self.k_values, ps, 'std', bins=nBins)
        errors = stds / np.sqrt(counts)
        return means, kBins, errors