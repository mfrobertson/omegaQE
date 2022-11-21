import numpy as np
from fields import Fields
from fisher import Fisher
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmology import Cosmology
from powerspectra import Powerspectra
import copy

class OmegaQE:

    def __init__(self, field_labels, exp="SO", N_pix=2 ** 7, Lmax=5000, kappa_map=None):
        self._power = Powerspectra()
        self._cosmo = Cosmology()
        self._fish = Fisher()
        self.fields = Fields(field_labels, exp=exp, N_pix=N_pix, kmax=Lmax, kappa_map=kappa_map)
        self._L_map = self.fields.kM
        self._Lx_map, self._Ly_map = self.get_Lx_Ly_maps()
        self.F_L_spline, self.C_inv_spline = self._get_F_L_and_C_inv_splines(Lmax)
        self._Cls = dict.fromkeys(self.fields.fields)
        self._windows = dict.fromkeys(self.fields.fields)

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

    def _populate_Cls_and_windows(self, Chis):
        for field in self.fields.fields:
            self._Cls[field], self._windows[field] = self._get_ps_and_window(Chis, None, field)
        if "k" not in self.fields.fields:
            self._Cls["k"], self._windows["k"] = self._get_ps_and_window(Chis, None, "k")

    def _get_ps_and_window(self, Chis, Chi_source2, field, nu=353e9, gal_distro="LSST_gold"):
        shape_1, shape_2 = np.shape(self._L_map)
        if field == "k":
            Cl = self._power.get_kappa_ps_2source(self._L_map.flatten(), Chis, Chi_source2)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.cmb_lens_window_matter(Chis, self._cosmo.get_chi_star())
            return Cl, window
        if field == "g":
            Cl = self._power.get_gal_kappa_ps(self._L_map.flatten(), Chis, gal_distro=gal_distro)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.gal_window_Chi(Chis)
            return Cl, window
        if field == "I":
            Cl = self._power.get_cib_kappa_ps(self._L_map.flatten(), nu=nu, Chi_source1=Chis)
            Cl = np.reshape(Cl, (shape_1, shape_2, np.size(Chis)))
            window = self._cosmo.cib_window_Chi(Chis, nu)
            return Cl, window

    def get_Lx_Ly_maps(self):
        Lx, Ly = self.fields.get_kx_ky(self.fields.N_pix, self.fields.get_dist() / self.fields.N_pix)
        Lx_map = Lx[None, :] * np.ones(np.shape(self.fields.kM))
        Ly_map = Ly[:, None] * np.ones(np.shape(self.fields.kM))
        return Lx_map, Ly_map

    def L_comp_map(self, index):
        if index == 0:
            return self._Lx_map
        if index == 1:
            return self._Ly_map

    def _get_matter_ps(self, Chis):
        Chis = Chis[None, None, :]
        Ls = self._L_map[:, :, None]
        zs = self._cosmo.Chi_to_z(Chis)
        ks = Ls / Chis
        matter_PK = self._cosmo.get_matter_PK(typ="matter")
        matter_ps = self._cosmo.get_matter_ps(matter_PK, zs, ks, weyl_scaled=False, typ="matter")
        return matter_ps

    def get_L_fac(self, p, r):
        L_p = self.L_comp_map(p)
        L_r = self.L_comp_map(r)
        return L_p * L_r / ((self._L_map + 0.5) ** 2)

    def get_f_g(self, p, r, q, s, Chi_i, noise):
        L_fac_f = self.get_L_fac(p, r)
        L_fac_g = self.get_L_fac(q, s)
        C_inv_spline = self.C_inv_spline
        h_f = 0
        h_g = 0
        for iii, field_i in enumerate(self.fields.fields):
            for jjj, field_j in enumerate(self.fields.fields):
                a_j = copy.deepcopy(self.fields.fft_maps[field_j])
                if noise:
                    a_j += copy.deepcopy(self.fields.fft_noise_maps[field_j])
                h_f += self._windows[field_i][Chi_i] * a_j * C_inv_spline[iii, jjj](self._L_map)
                h_g += self._Cls[field_i][:, :, Chi_i] * a_j * C_inv_spline[iii, jjj](self._L_map)
        return L_fac_f * h_f * self.matter_ps[:, :, Chi_i], L_fac_g * h_g

    def get_omega(self, Nchi=100, noise=False):
        Chis = np.linspace(0, self._cosmo.get_chi_star(), Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        self._populate_Cls_and_windows(Chis)
        self.matter_ps = self._get_matter_ps(Chis)
        norm = "ortho"
        I = np.zeros((np.shape(self._L_map)), dtype="complex128")
        r = 1
        s = 0
        for Chi_i, Chi in enumerate(Chis):
            I_tmp = np.zeros((np.shape(self._L_map)), dtype="complex128")
            for p in range(2):
                q = p
                f_i, g_j = self.get_f_g(p, r, q, s, Chi_i, noise)
                f_j, g_i = self.get_f_g(q, s, p, r, Chi_i, noise)
                F_i = np.fft.irfft2(f_i, norm=norm)*(2*np.pi)**2
                G_j = np.fft.irfft2(g_j, norm=norm)*(2*np.pi)**2
                F_j = np.fft.irfft2(f_j, norm=norm)*(2*np.pi)**2
                G_i = np.fft.irfft2(g_i, norm=norm)*(2*np.pi)**2
                I_tmp += (2*np.pi)**2 * 2 * np.fft.rfft2((F_i * G_j) - (F_j * G_i), norm=norm)
            I += I_tmp / (Chi ** 2) * self._windows["k"][Chi_i] * dChi / (2*np.pi)**2
        return I / self.F_L_spline(self._L_map)
