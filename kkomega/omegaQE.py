import numpy as np
from fisher import Fisher
from fields import Fields
from powerspectra import Powerspectra
from cosmology import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline


class OmegaQE:

    def __init__(self, field_labels, N_pix=2 ** 7, Lmax=5000, kappa_map=None):
        self._fields = Fields(field_labels, N_pix, Lmax, kappa_map)
        self.L_map = self._fields.kM
        self.Lx_map, self.Ly_map = self._get_Lx_Ly_maps()
        self.Cls = dict.fromkeys(self._fields.fields)
        self.windows = dict.fromkeys(self._fields.fields)
        self.a_bars = dict.fromkeys(self._fields.fields)
        self._power = Powerspectra()
        self._cosmo = Cosmology()
        self._fish = Fisher()
        self.F_L_spline, self.C_inv_spline = self._get_F_L_and_C_inv_splines(Lmax)
        self.populate_a_bars()

    def _populate_Cls_and_windows(self, Chis):
        for field in self._fields.fields:
            self.Cls[field], self.windows[field] = self._get_ps_and_window(self.L_map, Chis, self._cosmo.get_chi_star(), field)

    def _get_ps_and_window(self, ells, Chis, Chi_source2, field, nu=353e9, gal_distro="LSST_gold"):
        shape_1, shape_2 = np.shape(self.L_map)
        if field == "k":
            Cl = self._power.get_kappa_ps_2source(ells.flatten(), Chis, Chi_source2)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.cmb_lens_window(Chis, Chi_source2)
            return Cl, window
        if field == "g":
            Cl = self._power.get_gal_kappa_ps(ells.flatten(), Chis, gal_distro=gal_distro)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.gal_window_Chi(Chis)
            return Cl, window
        if field == "I":
            Cl = self._power.get_cib_kappa_ps(ells.flatten(), nu=nu, Chi_source1=Chis)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.cib_window_Chi(Chis, nu)
            return Cl, window

    def _get_Lx_Ly_maps(self):
        Lx, Ly = self._fields.get_kx_ky(self._fields.N_pix, self._fields.get_dist() / self._fields.N_pix)
        Lx_map = Lx[None, :] * np.ones(np.shape(self._fields.kM))
        Ly_map = Ly[:, None] * np.ones(np.shape(self._fields.kM))
        return Lx_map, Ly_map

    def _L_comp_map(self, index):
        if index == 0:
            return self.Lx_map
        if index == 1:
            return self.Ly_map

    def _get_f(self, i, p, r, weyl_ps, matter_weyl_ps, Chi_index):
        window = self.windows[i][Chi_index]
        L_map = self.L_map
        if i == "k":
            h = (L_map + 0.5) ** 4 * window * weyl_ps[:, :, Chi_index]
        else:
            h = -(L_map + 0.5) ** 2 * window * matter_weyl_ps[:, :, Chi_index]
        return self._L_comp_map(p) * self._L_comp_map(r) / ((L_map + 0.5) ** 2) * h

    def _get_g(self, i, p, r, Chi_index):
        return self._L_comp_map(p) * self._L_comp_map(r) / ((self.L_map + 0.5) ** 2) * self.Cls[i][:, :, Chi_index]

    def _get_matter_ps(self, Chis):
        Chis = Chis[None, None, :]
        Ls = self.L_map[:, :, None]
        zs = self._cosmo.Chi_to_z(Chis)
        ks = (Ls + 0.5) / Chis
        weyl_PK = self._cosmo.get_matter_PK(typ="weyl")
        weyl_ps = self._cosmo.get_matter_ps(weyl_PK, zs, ks, weyl_scaled=False, typ="weyl")
        matter_weyl_PK = self._cosmo.get_matter_PK(typ="matter-weyl")
        matter_weyl_ps = self._cosmo.get_matter_ps(matter_weyl_PK, zs, ks, weyl_scaled=False, typ="matter-weyl")
        return weyl_ps, matter_weyl_ps

    def _get_F_L_and_C_inv_splines(self, Lmax=5000):
        sample_Ls = self._fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=Lmax, Nells=50)
        sample_Ls, F_L, C_inv = self._fish.get_F_L(self._fields.fields, Ls=sample_Ls, Ntheta=100, nu=353e9, return_C_inv=True)
        F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        N_fields = np.size(self._fields.fields)
        C_inv_splines = np.empty((N_fields, N_fields), dtype=InterpolatedUnivariateSpline)
        Ls = np.arange(Lmax + 1)
        for iii in range(N_fields):
            for jjj in range(N_fields):
                C_inv_ij = C_inv[iii, jjj]
                C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[2:], C_inv_ij[2:])
        return F_L_spline, C_inv_splines

    def populate_a_bars(self):
        Ls = self.L_map.flatten()
        N_fields = np.size(self._fields.fields)
        a = np.zeros((np.size(Ls), N_fields, 1))
        C_inv = np.zeros((np.size(Ls), N_fields, N_fields))
        for iii, field_i in enumerate(self._fields.fields):
            a[:, iii, 0] = self._fields.fft_maps[field_i].flatten()
            for jjj, field_j in enumerate(self._fields.fields):
                C_inv[:, iii, jjj] = self.C_inv_spline[iii][jjj](Ls)
        a_bars = np.matmul(C_inv, a)
        for iii, field in enumerate(self._fields.fields):
            a_bar = a_bars[:, iii, 0]
            self.a_bars[field] = np.reshape(a_bar, np.shape(self.L_map))

    def get_omega(self, Nchi=100):
        Chis = np.linspace(0, self._cosmo.get_chi_star(), Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        weyl_ps, matter_weyl_ps = self._get_matter_ps(Chis)
        self._populate_Cls_and_windows(Chis)
        r_s = [(0, 1), (1, 0)]
        I = np.zeros((self._fields.N_pix, self._fields.N_pix, np.size(Chis)))
        for Chi_index, Chi in enumerate(Chis):
            for i in self._fields.fields:
                for j in self._fields.fields:
                    a_i = self.a_bars[i]
                    a_j = self.a_bars[j]
                    for p in range(2):
                        q = p
                        for r, s in r_s:
                            fac = -1 if r > s else 1
                            F_i_fft = fac * self._get_f(i, p, r, weyl_ps, matter_weyl_ps, Chi_index) * a_i
                            G_j_fft = fac * self._get_g(j, q, s, Chi_index) * a_j
                            F_j_fft = fac * self._get_f(j, q, s, weyl_ps, matter_weyl_ps, Chi_index) * a_j
                            G_i_fft = fac * self._get_g(i, p, r, Chi_index) * a_i

                            F_i = np.fft.irfft2(F_i_fft, norm="ortho")
                            F_j = np.fft.irfft2(F_j_fft, norm="ortho")
                            G_i = np.fft.irfft2(G_i_fft, norm="ortho")
                            G_j = np.fft.irfft2(G_j_fft, norm="ortho")
                            I[:, :, Chi_index] += (F_i * G_j) - (F_j * G_i)

        omega_F = dChi * np.sum(I * self.windows["k"] / (Chis ** 2), axis=2)
        return np.fft.rfft2(omega_F, norm="ortho") / self.F_L_spline(self.L_map)

