import numpy as np
from fields import Fields
from fisher import Fisher
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmology import Cosmology
from powerspectra import Powerspectra

class OmegaQE:

    def __init__(self, field_labels, exp="SO", N_pix=2 ** 7, Lmax=5000, kappa_map=None):
        self._power = Powerspectra()
        self._cosmo = Cosmology()
        self._fish = Fisher()
        self.N_pix = N_pix
        self.fields = Fields(field_labels, exp, self.N_pix, Lmax, kappa_map)
        self.L_map = self.fields.kM
        self.Lx_map, self.Ly_map = self.get_Lx_Ly_maps()
        self.F_L_spline, self.C_inv_spline = self._get_F_L_and_C_inv_splines(Lmax)
        self.matter_PK = self._cosmo.get_matter_PK(typ="matter")

    def _get_F_L_and_C_inv_splines(self, Lmax=5000):
        sample_Ls = self._fish.covariance.get_log_sample_Ls(Lmin=2, Lmax=Lmax, Nells=150)
        sample_Ls, F_L, C_inv = self._fish.get_F_L(self.fields.fields, Ls=sample_Ls, Ntheta=100, nu=353e9, return_C_inv=True)
        F_L_spline = InterpolatedUnivariateSpline(sample_Ls, F_L)
        N_fields = np.size(self.fields.fields)
        C_inv_splines = np.empty((N_fields, N_fields), dtype=InterpolatedUnivariateSpline)
        Ls = np.arange(Lmax + 1)
        for iii in range(N_fields):
            for jjj in range(N_fields):
                C_inv_ij = C_inv[iii, jjj]
                C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[2:], C_inv_ij[2:])
        return F_L_spline, C_inv_splines

    def _get_Cl_and_window(self, Chi, field, nu=353e9, gal_distro="LSST_gold"):
        if field == "k":
            Cl = self._power.get_kappa_ps_2source(self.L_map.flatten(), Chi, self._cosmo.get_chi_star(), use_weyl=False)
            Cl = np.reshape(Cl, np.shape(self.L_map))
            window = self._cosmo.cmb_lens_window_matter(Chi, self._cosmo.get_chi_star(), False)
            return Cl, window
        if field == "g":
            Cl = self._power.get_gal_kappa_ps(self.L_map.flatten(), Chi, gal_distro=gal_distro, use_weyl=False)
            Cl = np.reshape(Cl, np.shape(self.L_map))
            window = self._cosmo.gal_window_Chi(Chi)
            return Cl, window
        if field == "I":
            Cl = self._power.get_cib_kappa_ps(self.L_map.flatten(), nu=nu, Chi_source1=Chi, use_weyl=False)
            Cl = np.reshape(Cl, np.shape(self.L_map))
            window = self._cosmo.cib_window_Chi(Chi, nu)
            return Cl, window

    def get_Lx_Ly_maps(self):
        Lx, Ly = self.fields.get_kx_ky()
        Lx_map = Lx[None, :] * np.ones(np.shape(self.L_map))
        Ly_map = Ly[:, None] * np.ones(np.shape(self.L_map))
        return Lx_map, Ly_map

    def L_comp_map(self, index):
        if index == 0:
            return self.Lx_map
        if index == 1:
            return self.Ly_map

    def get_L_fac(self, p, r):
        L_p = self.L_comp_map(p)
        L_r = self.L_comp_map(r)
        return L_p * L_r / ((self.L_map + 0.5) ** 2)

    def get_f_g(self, p, r, q, s, Cls, windows, matter_ps, noise):
        L_fac_f = self.get_L_fac(p, r)
        L_fac_g = self.get_L_fac(q, s)
        C_inv_spline = self.C_inv_spline
        h_f = 0
        h_g = 0
        for iii, field_i in enumerate(self.fields.fields):
            for jjj, field_j in enumerate(self.fields.fields):
                if noise:
                    a_j = self.fields.fft_maps[field_j] + self.fields.fft_noise_maps[field_j]
                else:
                    a_j = self.fields.fft_maps[field_j]
                h_f += windows[field_i] * a_j * C_inv_spline[iii, jjj](self.L_map)
                h_g += Cls[field_i] * a_j * C_inv_spline[iii, jjj](self.L_map)
        return L_fac_f * h_f * matter_ps, L_fac_g * h_g

    def _get_matter_ps(self, Chi):
        z = self._cosmo.Chi_to_z(Chi)
        ks = (self.L_map + 0.5) / Chi
        return self._cosmo.get_matter_ps(self.matter_PK, z, ks, weyl_scaled=False, typ="matter")

    def get_omega(self, Nchi=20, noise=True):
        norm="forward"
        Lx, Ly = self.fields.get_kx_ky()
        dx = Lx[1] - Lx[0]
        dy = Ly[1] - Ly[0]
        Chis = np.linspace(0, self._cosmo.get_chi_star(), Nchi + 1)[1:]
        dChi = Chis[1] - Chis[0]
        I = np.zeros((np.shape(self.L_map)), dtype="complex128")
        r = 1
        s = 0
        for Chi_i, Chi in enumerate(Chis):
            Cls = dict.fromkeys(self.fields.fields)
            windows = dict.fromkeys(self.fields.fields)
            matter_ps = self._get_matter_ps(Chi)
            for field in self.fields.fields:
                Cls[field], windows[field] = self._get_Cl_and_window(Chi, field)
            I_tmp = np.zeros((np.shape(self.L_map)), dtype="complex128")
            for p in range(2):
                q = p
                f_i, g_j = self.get_f_g(p, r, q, s, Cls, windows, matter_ps, noise)
                f_j, g_i = self.get_f_g(q, s, p, r, Cls, windows, matter_ps, noise)
                F_i = np.fft.irfft2(f_i, norm=norm)
                G_j = np.fft.irfft2(g_j, norm=norm)
                F_j = np.fft.irfft2(f_j, norm=norm)
                G_i = np.fft.irfft2(g_i, norm=norm)
                I_tmp += 2 * np.fft.rfft2((F_i * G_j) - (F_j * G_i), norm=norm)
            I += I_tmp / (Chi ** 2) * windows['k']
        return I * dChi * dx * dy / self.F_L_spline(self.L_map) / ((2 * np.pi) ** 2)
