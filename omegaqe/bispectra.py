from omegaqe.modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum to leading order in the Post Born approximation.

    """

    class M_spline:


        def __init__(self, spline, nu, gal_bins, gal_distro="LSST_gold"):
            self.spline = spline
            self.nu = nu
            self.gal_bins = gal_bins
            self.gal_distro = gal_distro

    def __init__(self, powerspectra=None):
        """
        Constructor.

        Parameters
        ----------
        M_spline : bool
            On instantiation, build a spline for estimation of the mode-coupling components which can be used for quicker calculation.
        ells_sample : ndarray
            1D array of sample multipole moments. If not M_matrix is supplied these will be used for generating the spline.
        M_matrix : ndarray
            2D array of the modecoupling matrix at calculated at the positions given by ells_sample.
        """
        self._mode = Modecoupling(powerspectra=powerspectra)
        self.init_M_splines()

    def init_M_splines(self):
        self._M_splines = dict.fromkeys(self._mode.get_M_types())
        self._M_splines_lens_delta = {
            "g": dict.fromkeys(self._mode.get_M_types()),
            "I": dict.fromkeys(self._mode.get_M_types()),
            "a": dict.fromkeys(self._mode.get_M_types())
        }

    def _triangle_dot_product(self, mag1, mag2, mag3):
        res = (-(mag1**2) - (mag2**2) + (mag3**2))/2
        # res[np.isnan(res)] = 0  # tmp
        return res

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        res = -2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))     # I think sign here is arbitrary?
        # res[np.isnan(res)] = 0   # tmp
        return res

    def _bispectra_prep(self, typ,  L1, L2, L3=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold"):
        sec_var = typ[-1]
        L12_dot = None
        if L3 is not None:
            L12_dot = self._triangle_dot_product(L1, L2, L3)
        M_typ1 = typ[:2]
        M_typ2 = M_typ1[::-1]
        if M_spline:
            if sec_var in ("k", "w"):
                M_spline_cache = self._M_splines
            elif sec_var in ("g", "I", "a"):
                M_spline_cache = self._M_splines_lens_delta[sec_var]
            else:
                raise ValueError(f"Unrecognized value for sec_var: {sec_var}")
            M1 = M_spline_cache[M_typ1].spline.ev(L1, L2)
            M2 = M_spline_cache[M_typ2].spline.ev(L2, L1)
            return M1, M2, L12_dot
        raise ValueError("Don't want code to reach here, always use spline...")
        M1 = self._mode.components(L1, L2, typ=M_typ1, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        M2 = self._mode.components(L2, L1, typ=M_typ2, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        return M1, M2, L12_dot

    def _build_M_spline(self, typ, ells_sample, M_matrix, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold", sec_var="k"):
        if ells_sample is not None and M_matrix is not None:
            spline = self._mode.spline(ells_sample, M_matrix, typ=typ, gal_distro=gal_distro, sec_order_var=sec_var)
            return self.M_spline(spline, nu, gal_bins, gal_distro=gal_distro)
        if ells_sample is None:
            spline = self._mode.spline(typ=typ, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro, sec_order_var=sec_var)
            return self.M_spline(spline, nu, gal_bins, gal_distro=gal_distro)
        spline = self._mode.spline(ells_sample, typ=typ, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro, sec_order_var=sec_var)
        return self.M_spline(spline, nu, gal_bins, gal_distro=gal_distro)

    def build_M_spline(self, typ="kk", ells_sample=None, M_matrix=None, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None)):
        """
        Generates and stores/replaces spline for the mode-coupling matrix.

        ells_sample : ndarray
            1D array of sample multipole moments. If not M_matrix is supplied these will be used for generating the spline.
        M_matrix : ndarray
            2D array of the modecoupling matrix at calculated at the positions given by ells_sample.

        Returns
        -------
        RectBivariateSpline
            Returns the resulting spline object of the mode coupling matrix. RectBivariateSpline(ells1,ells2) will produce matrix. RectBivariateSpline.ev(ells1,ells2) will calculate components of the matrix.
        """
        if not self._mode.check_type(typ):
            raise ValueError(f"Modecoupling type {typ} does not exist")
        self._M_splines[typ] = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax, nu, gal_bins)

    def _build_M_splines_lens_delta(self, typ, nu, gal_bins, gal_distro="LSST_gold"):
        typs = list(typ)
        for sec_var in typs:
            if sec_var != "k":
                M_typ1 = typ.replace(sec_var, '', 1)
                M_typ2 = M_typ1[::-1]
                M_spline_ld = self._M_splines_lens_delta[sec_var][M_typ1]
                if M_spline_ld is not None:
                    if (M_spline_ld.nu == nu) and (M_spline_ld.gal_bins == gal_bins) and (M_spline_ld.gal_distro == gal_distro):
                        continue
                print(f"Building M-spline for {sec_var} lens delta matrices for {M_typ1}.")
                self._M_splines_lens_delta[sec_var][M_typ1] = self._build_M_spline(M_typ1, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro, sec_var=sec_var)
                if M_typ2 != M_typ1:
                    print(f"Building M-spline for {sec_var} lens delta matrices for {M_typ2}.")
                    self._M_splines_lens_delta[sec_var][M_typ2] = self._build_M_spline(M_typ2, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro, sec_var=sec_var)

    def _build_M_splines(self, typ, nu, gal_bins, gal_distro="LSST_gold"):
        M_typ1 = typ[:2]
        M_typ2 = M_typ1[::-1]
        if self._M_splines[M_typ1] is not None:
            if self._M_splines[M_typ1].nu != nu or self._M_splines[M_typ1].gal_bins != gal_bins or self._M_splines[M_typ1].gal_distro != gal_distro:
                self._M_splines[M_typ1] = self._build_M_spline(M_typ1, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro)
                if M_typ2 != M_typ1:
                    self._M_splines[M_typ2] = self._build_M_spline(M_typ2, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro)
            return
        self._M_splines[M_typ1] = self._build_M_spline(M_typ1, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro)
        if M_typ2 != M_typ1:
            self._M_splines[M_typ2] = self._build_M_spline(M_typ2, None, None, 0, None, nu, gal_bins, gal_distro=gal_distro)

    def _check_type(self, typ, pb=True):
        typs = self._mode.get_M_types()
        sec_vars = ("w", "k") if pb else ("g", "I")
        if (typ[:-1] not in typs) or (typ[-1] not in sec_vars):
            raise ValueError(f"Bispectrum type {typ} not from accepted types: {typs}")

    def check_type(self, typ):
        """


        Parameters
        ----------
        typ

        Returns
        -------

        """
        try:
            self._check_type(typ)
        except:
            return False
        return True

    def _pb_bispectrum(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold"):
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        if typ[-1] == "w":
            product_func = self._triangle_cross_product
            fac = -1
        elif typ[-1] == "k":
            product_func = self._triangle_dot_product
            fac = +1
        else:
            raise ValueError(f"Bispectrum type {typ} has second order variable {typ[-1]} not from expected 'k' or 'w'.")
        L13_fac = product_func(L1, L3, L2)
        L23_fac = fac * product_func(L2, L3, L1)
        res = 2 * L12_dot * ((L13_fac * M1) + (L23_fac * M2))/(L1**2 * L2**2)
        if np.size(res) == 1:
            if np.isnan(res):
                return 0
            return res
        res[np.isnan(res)] = 0
        return res

    def _omega_bispectrum_angle(self, typ, L1, L2, theta12, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold"):
        M1, M2, _ = self._bispectra_prep(typ, L1, L2, None, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        return np.sin(2 * theta12) * (M1 - M2)    #Using clockwise omega convention of Pratten & Lewis

    def _get_F2(self, k1, k2, k3):
        A = 1
        B = 1
        C = 1
        k12_dot = self._triangle_dot_product(k1, k2, k3)
        A_fac= 5/7
        B_fac = k12_dot/(2*k1*k2) * ((k1/k2) + (k2/k1))
        C_fac = 2/7 * (k12_dot/(k1*k2))**2
        return (A_fac * A) + (B_fac * B) + (C_fac * C)

    def _get_matter_ps(self, L, chi, extended=False):
        z = self._mode._cosmo.Chi_to_z(chi)
        k = (L + 0.5)/ chi if extended else L / chi
        w = np.ones(np.shape(k))
        w[k < 1e-4] = 0
        w[k >= 100] = 0
        return w*self._mode._cosmo.get_matter_ps(self._mode.matter_PK, z, k, curly=False, weyl_scaled=False, typ="matter")

    def _vectorise_ells(self, ells):
        if np.size(ells) == 1:
            return ells
        if ells.ndim == 1:
            return ells[:, None]
        if ells.ndim == 2:
            return ells[:, :, None]
        else:
            raise ValueError(f"Too many (or too few) dimensions {ells.ndim}")

    def _delta_bispectrum(self, L1, L2, L3, chi, extended=False):
        L1 = self._vectorise_ells(L1)
        L2 = self._vectorise_ells(L2)
        L3 = self._vectorise_ells(L3)
        k1 = (L1 + 0.5) / chi if extended else L1 / chi
        k2 = (L2 + 0.5) / chi if extended else L2 / chi
        k3 = (L3 + 0.5) / chi if extended else L3 / chi
        matter_ps1 = self._get_matter_ps(L1, chi, extended)
        matter_ps2 = self._get_matter_ps(L2, chi, extended)
        matter_ps3 = self._get_matter_ps(L3, chi, extended)
        bi1 = 2 * self._get_F2(k1, k2, k3) * matter_ps1 * matter_ps2
        bi2 = 2 * self._get_F2(k2, k3, k1) * matter_ps2 * matter_ps3
        bi3 = 2 * self._get_F2(k3, k1, k2) * matter_ps3 * matter_ps1
        return bi1 + bi2 + bi3

    def get_lss_bispectrum(self, typ, L1, L2, L3=None, theta=None, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold"):
        if "w" in typ:
            return 0
        if L3 is None:
            return self.get_lss_bispectrum(typ, L1, L2, self._get_third_L(L1, L2, theta), None, zmin, zmax, nu, gal_bins, gal_distro)
        Nchi = 100
        _, chis, dChi, win1, win2 = self._mode._integral_prep(Nchi, zmin, zmax, typ[:-1], nu, gal_bins, gal_distro=gal_distro, sec_order_var=typ[1])
        win3 = self._mode._get_window(typ[2], chis, nu, gal_bins, gal_distro)
        return np.sum(win1 * win2 * win3 * self._delta_bispectrum(L1, L2, L3, chis) / (chis ** 4), axis=-1) * dChi

    def _lens_delta_bispectrum(self, typ, L1, L2, L3, M_spline, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold"):
        if "w" in typ:
            return 0
        if typ[-1] == "k":
            raise ValueError("Lens delta bispectrum type has second order variable 'k'?")
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        res = -2 * L12_dot * ((M1 / L2**2) + (M2 / L1**2))
        if np.size(res) == 1:
            if np.isnan(res):
                return 0
            return res
        res[np.isnan(res)] = 0
        return res

    def _get_lens_delta_bis(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold", verbose=False):
        if "w" in typ or typ == "kkk":
            return 0
        if typ[-1] != "k":
            raise ValueError(f"Something has gone wrong...")
        if M_spline:
            self._build_M_splines_lens_delta(typ, nu, gal_bins, gal_distro=gal_distro)
        if "kk" in ''.join(sorted(typ)):
            if typ[0] == "k":  # kak
                if verbose: print(f"Including lensed delta {'kk'+typ[1]} bispectrum")
                return self._lens_delta_bispectrum("kk"+typ[1], L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            else:  # akk
                if verbose: print(f"Including lensed delta {'kk' + typ[0]} bispectrum")
                return self._lens_delta_bispectrum("kk"+typ[0], L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
        if "k" in typ:
            if verbose: print(f"Including lensed delta {'k' + typ[1] + typ[0]} and {typ[0] + 'k' + typ[1]} bispectra")
            lens_bi1 = self._lens_delta_bispectrum("k" + typ[1] + typ[0], L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            lens_bi2 = self._lens_delta_bispectrum(typ[0] + "k" + typ[1], L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            return lens_bi1 + lens_bi2
        print("Shouldn't see me in fisher or F_L calc...")
        if verbose: print(f"Including lensed delta {typ} and {typ[0] + typ[2] + typ[1]} and {typ[2] + typ[1] + typ[0]} bispectra")
        lens_bi1 = self._lens_delta_bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins,gal_distro=gal_distro)
        lens_bi2 = self._lens_delta_bispectrum(typ[0] + typ[2] + typ[1], L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins,gal_distro=gal_distro)
        lens_bi3 = self._lens_delta_bispectrum(typ[2] + typ[1] + typ[0], L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins,gal_distro=gal_distro)
        return lens_bi1 + lens_bi2 + lens_bi3

    def get_ld_bispectrum(self, typ, L1, L2, L3=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold", verbose=False):
        # self._check_type(typ, pb=False)
        self._check_type(typ)
        # if M_spline:
        #     self._build_M_splines_lens_delta(typ, nu, gal_bins, gal_distro=gal_distro)
        # return self._lens_delta_bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro)
        return self._get_lens_delta_bis(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro, verbose)

    def _get_third_L(self, L1, L2, theta):
        # Using cosine rule (remember that theta is not same as internal angle of bispectrum traingle)
        return np.sqrt(L1 ** 2 + L2 ** 2 + (2 * L1 * L2 * np.cos(theta).astype("double"))).astype("double")

    def _get_ll_bispectrum(self, typ, L1, L2, L3, M_spline=False, zmin=0, zmax=None):
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax)
        res = -2 * L12_dot**2 * (M1 + M2) / (L1 ** 2 * L2 ** 2)
        res[np.isnan(res)] = 0
        return res

    def get_ll_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        if L3 is None:
            L3 = self._get_third_L(L1, L2, theta)
        return self._ll_rd_bispectrum("ll", typ, L1, L2, L3, M_spline, zmin, zmax)

    def _get_rd_bispectrum(self, typ, L1, L2, L3, M_spline=False, zmin=0, zmax=None):
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax)
        res = -2 * L12_dot * ((L1**2 * M1) + (L2**2 * M2)) / ( L1 ** 2 * L2 ** 2)
        res[np.isnan(res)] = 0
        return res

    def get_rd_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        if L3 is None:
            L3 = self._get_third_L(L1, L2, theta)
        return self._ll_rd_bispectrum("rd", typ, L1, L2, L3, M_spline, zmin, zmax)

    def _ll_rd_bispectrum(self, term, typ, L1, L2, L3, M_spline, zmin, zmax):
        if term == "ll":
            bi_func = self._get_ll_bispectrum
        elif term == "rd":
            bi_func = self._get_rd_bispectrum
        else:
            raise ValueError(f"Unknown term {term}")
        b = bi_func(typ, L1, L2, L3, M_spline, zmin, zmax)
        if typ[:2] == "kk":
            b += bi_func(typ, L3, L2, L1, M_spline, zmin, zmax)
            b += bi_func(typ, L1, L3, L2, M_spline, zmin, zmax)
        elif "k" in typ[:2]:
            if "typ"[0] == "k":
                b += bi_func(typ, L3, L2, L1, M_spline, zmin, zmax)
            else:
                b += bi_func(typ, L1, L3, L2, M_spline, zmin, zmax)
        return b

    def get_pb_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold", verbose=False, one_perm=False):
        self._check_type(typ)
        if M_spline:
            self._build_M_splines(typ, nu, gal_bins, gal_distro=gal_distro)
        if L3 is not None:
            b = self._pb_bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            if one_perm:
                if verbose: print(f"Including one permutation of pB bispectrun {typ}")
                return b
            if typ == "kkk":
                if verbose: print(f"Including all three pB bispectra of {typ}")
                b += self._pb_bispectrum(typ, L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
                b += self._pb_bispectrum(typ, L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            elif "kk" in ''.join(sorted(typ)) and "w" not in typ:
                if typ[0] == "k":   #kak
                    if verbose: print(f"Including all two pB bispectra of {typ} (kak)")
                    b += self._pb_bispectrum(typ, L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
                else:   #akk
                    if verbose: print(f"Including all two pB bispectra of {typ} (akk)")
                    b += self._pb_bispectrum(typ, L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            else:
                if verbose: print(f"Including all one pB bispectra of {typ} (abk)")
            return b
        if typ[-1] == "w":
            return self._omega_bispectrum_angle(typ, L1, L2, theta, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
        return self.get_pb_bispectrum(typ, L1, L2, self._get_third_L(L1, L2, theta), None, M_spline, zmin, zmax, nu, gal_bins, gal_distro,verbose=verbose, one_perm=one_perm)


    def get_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold", lens_delta=False, include_lss=False, verbose=False, one_perm=False):
        """
        Calculates cmb lensing bispectrum for the combination of observables specified.

        Parameters
        ----------
        L1 : int or float or ndarray
            Magnitude(s) of the first multiple moment. Must be of same dimensions as other moments.
        L2 : int or float or ndarray
            Magnitude(s) of the second multiple moment. Must be of same dimensions as other moments.
        L3 : int or float or ndarray
            Magnitude(s) of the third multiple moment. Must be of same dimensions as other moments.
        M_spline : bool
            Use an interpolated estimation of the mode-coupling matrix for quicker computation.

        Returns
        -------
        float or ndarray
            The bispectrum.
        """
        sec_var = typ[-1]
        b = 0
        if sec_var in ("w", "k"):
            if verbose: print("Including post born bispectra terms")
            b += self.get_pb_bispectrum(typ, L1, L2, L3, theta, M_spline, zmin, zmax, nu, gal_bins, gal_distro, verbose, one_perm=one_perm)
        elif sec_var == "L":
            if verbose: print("Including lens-lens post born bispectra terms")
            typ = typ.replace(sec_var, "k")
            b += self.get_ll_bispectrum(typ, L1, L2, L3, theta, M_spline, zmin, zmax)
        elif sec_var == "D":
            if verbose: print("Including ray-deflect post born bispectra terms")
            typ = typ.replace(sec_var, "k")
            b += self.get_rd_bispectrum(typ, L1, L2, L3, theta, M_spline, zmin, zmax)
        else:
            raise ValueError("Sec var must be either 'w', 'k', 'D', or 'L'")
        if L3 is None:
            L3 = self._get_third_L(L1, L2, theta)
        if lens_delta:
            if verbose: print("Including lensed delta bispectra terms")
            b += self._get_lens_delta_bis(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro, verbose)
        if include_lss:
            if verbose: print("Including lss bispectrum term")
            b += self.get_lss_bispectrum(typ, L1, L2, L3, None, zmin, zmax, nu, gal_bins, gal_distro)
        return b

