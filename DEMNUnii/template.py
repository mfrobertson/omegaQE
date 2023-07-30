import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import datetime
from fields import Fields
import healpy as hp
import omegaqe




class Template:

    def __init__(self, fields: Fields, Lmin=30, Lmax=3000, F_L_spline=None, C_inv_splines=None, tracer_noise=False, use_kappa_rec=False, kappa_rec_qe_typ="T", gaussCMB=False, diffCMBs=False, diffCMBs_offset=1):
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.fields = fields
        self.Lmax_map = fields.Lmax_map
        self.N_pix = fields.N_pix
        self._fish = self.fields.fish
        self._power = self._fish.power
        self._cosmo = self._power.cosmo
        self.F_L, self.C_inv = self._get_F_L_and_C_inv(F_L_spline, C_inv_splines)
        self.matter_PK = self._cosmo.get_matter_PK(typ="matter")
        self.a_bars = dict.fromkeys(self.fields.fields)
        self.use_kappa_rec = use_kappa_rec
        self.kappa_rec = self.fields.get_kappa_rec(kappa_rec_qe_typ)
        self._populate_a_bars(tracer_noise)

    def _get_F_L_and_C_inv(self, F_L_spline, C_inv_splines):
        Ls = np.load(f"{omegaqe.RESULTS_DIR}/F_L_results/{self.fields.fields}/{self.fields.exp}/gmv/TEB/30_3000/1_2000/Ls.npy")
        F_L = np.load(f"{omegaqe.RESULTS_DIR}/F_L_results/{self.fields.fields}/{self.fields.exp}/gmv/TEB/30_3000/1_2000/F_L.npy")
        C_inv = self.fields.fish.covariance.get_C_inv(self.fields.fields, self.Lmax_map, nu=353e9)
        F_L_spline = InterpolatedUnivariateSpline(Ls, F_L)
        Ls_sample = np.arange(self.Lmax_map+1)
        F_L = F_L_spline(Ls_sample)
        N_fields = len(self.fields.fields)
        C_invs = np.empty((N_fields, N_fields, np.size(F_L)))
        for iii in range(N_fields):
            for jjj in range(N_fields):
                C_inv_ij = C_inv[iii, jjj]
                C_inv_ij[self.Lmax + 1:] = 0
                C_inv_ij[:self.Lmin] = 0
                C_invs[iii, jjj] = C_inv_ij
        return F_L, C_invs

    def _get_fft_maps(self, field):
        if field == "k" and self.use_kappa_rec:
            return self.kappa_rec
        # if include_noise:
            # return self.fields.fft_maps[field] + self.fields.fft_noise_maps[field]
            pass
        return self.fields.fft_maps[field]

    def _populate_a_bars(self, tracer_noise):
        for iii, field_i in enumerate(self.fields.fields):
            a_bar_i = np.zeros(np.shape(self._get_fft_maps('k')), dtype="complex128")
            for jjj, field_j in enumerate(self.fields.fields):
                # a_j = self._get_fft_maps(field_j, tracer_noise)
                a_j = self._get_fft_maps(field_j)
                a_bar_i += hp.sphtfunc.almxfl(a_j, self.C_inv[iii, jjj])
            self.a_bars[field_i] = a_bar_i

    def _get_Cl_and_window(self, Chi, field, nu=353e9, gal_distro="LSST_gold"):
        Ls_sample = np.arange(1, self.Lmax_map + 1)
        Cl_sample = np.zeros(self.Lmax_map + 1)
        window = None
        if field == "k":
            Cl_sample[1:] = self._power.get_kappa_ps_2source(Ls_sample, Chi, self._cosmo.get_chi_star(), use_weyl=False)
            window = self._cosmo.cmb_lens_window_matter(Chi, self._cosmo.get_chi_star(), False)
        if field == "g":
            Cl_sample[1:] = self._power.get_gal_kappa_ps(Ls_sample, Chi, gal_distro=gal_distro, use_weyl=False)
            window = self._cosmo.gal_window_Chi(Chi)
        if field == "I":
            Cl_sample[1:] = self._power.get_cib_kappa_ps(Ls_sample, nu=nu, Chi_source1=Chi, use_weyl=False)
            window = self._cosmo.cib_window_Chi(Chi, nu)
        return Cl_sample, window

    def _get_window_k(self, Chi):
        return self._cosmo.cmb_lens_window_matter(Chi, self._cosmo.get_chi_star(), False)

    def _get_matter_ps(self, Chi):
        z = self._cosmo.Chi_to_z(Chi)
        ks_sample = np.arange(1, self.Lmax_map+1) / Chi   #This make sense in full sky?
        P_m = np.zeros(self.Lmax_map+1)
        P_m[1:] = self._cosmo.get_matter_ps(self.matter_PK, z, ks_sample, weyl_scaled=False, typ="matter")
        return P_m

    def _get_Egamma_Elambda(self, Cls, windows, matter_ps):
        # TODO: should I ensure that these are real_alms???? Or is this already achieved by a_bar?
        h_gamma = None
        h_lambda = None
        for iii, field_i in enumerate(self.fields.fields):
            if h_gamma is None:
                h_gamma = self.a_bars[field_i] * windows[field_i]
                h_lambda = hp.sphtfunc.almxfl(self.a_bars[field_i], Cls[field_i])
            else:
                h_gamma += self.a_bars[field_i] * windows[field_i]
                h_lambda += hp.sphtfunc.almxfl(self.a_bars[field_i], Cls[field_i])
        return hp.sphtfunc.almxfl(h_gamma, matter_ps), h_lambda

    def get_omega(self, Nchi=20):
        Chis = np.linspace(0, self._cosmo.get_chi_star(), Nchi + 1)[1:]
        dChi = Chis[1] - Chis[0]
        I_map_tot = None
        t0 = datetime.datetime.now()
        print(f"[00:00] {0}%", end='')
        for Chi_i, Chi in enumerate(Chis):
            Cls = dict.fromkeys(self.fields.fields)
            windows = dict.fromkeys(self.fields.fields)
            matter_ps = self._get_matter_ps(Chi)
            for field in self.fields.fields:
                Cls[field], windows[field] = self._get_Cl_and_window(Chi, field)

            E_gamma, E_lambda = self._get_Egamma_Elambda(Cls, windows, matter_ps)
            B_gamma = np.zeros(np.shape(E_gamma))
            B_lambda = np.zeros(np.shape(E_lambda))
            Q_gamma, U_gamma = hp.sphtfunc.alm2map_spin(np.array([E_gamma, B_gamma]), self.fields.dm.nside, 2, self.Lmax_map, self.Lmax_map)
            Q_lambda, U_lambda = hp.sphtfunc.alm2map_spin(np.array([E_lambda, B_lambda]), self.fields.dm.nside, 2, self.Lmax_map, self.Lmax_map)
            I_map = (Q_gamma * U_lambda) - (Q_lambda * U_gamma)

            window_k = self._get_window_k(Chi)
            if I_map_tot is None:
                I_map_tot = I_map / (Chi ** 2) * window_k
            else:
                I_map_tot += I_map / (Chi ** 2) * window_k
            print('\r', end='')
            print(f"[{str(datetime.datetime.now() - t0)[:-7]}] {int((Chi_i+1)/Nchi * 100)}%", end='')
        print("")
        I_alm = hp.sphtfunc.map2alm(I_map_tot, lmax=self.Lmax_map, mmax=self.Lmax_map, use_pixel_weights=False)
        return hp.sphtfunc.almxfl(I_alm, 2/self.F_L) * dChi
