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
            "I": dict.fromkeys(self._mode.get_M_types())
        }

    def _triangle_dot_product(self, mag1, mag2, mag3):
        res = (mag1**2 + mag2**2 - mag3**2)/2
        res[np.isnan(res)] = 0  # tmp
        return res

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        res = 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))
        res[np.isnan(res)] = 0   # tmp
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
            elif sec_var in ("g", "I"):
                M_spline_cache = self._M_splines_lens_delta[sec_var]
            else:
                raise ValueError(f"Unrecognized value for sec_var: {sec_var}")
            M1 = M_spline_cache[M_typ1].spline.ev(L1, L2)
            M2 = M_spline_cache[M_typ2].spline.ev(L2, L1)
            return M1, M2, L12_dot
        M1 = self._mode.components(L1, L2, typ=M_typ1, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        M2 = self._mode.components(L2, L1, typ=M_typ2, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        return M1, M2, L12_dot

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("kkk", L1, L2, L3, M_spline, zmin, zmax)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        res = 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)
        res[np.isnan(res)] = 0
        return res

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L1, L3, L2, M_spline, zmin, zmax)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L3, L2, L1, M_spline, zmin, zmax)

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
                    if (M_spline_ld.nu == nu) or (M_spline_ld.gal_bins == gal_bins) or (M_spline_ld.gal_distro != gal_distro):
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

    def _check_type(self, typ):
        typs = self._mode.get_M_types()
        if (typ[:-1] not in typs) or (typ[-1] not in ("w", "k")):
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

    def _lens_delta_bispectrum(self, typ, L1, L2, L3, M_spline, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold"):
        if "w" in typ:
            return 0
        # TODO: M1 and M2 are different for lens_delta
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        res = -2 * L12_dot * ((M1 / L2**2) + (M2 / L1**2))
        res[np.isnan(res)] = 0
        return res

    def _bispectrum(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold"):
        M1, M2, L12_dot = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        if typ[-1] == "w":
            product_func = self._triangle_cross_product
        elif typ[-1] == "k":
            product_func = self._triangle_dot_product
        else:
            raise ValueError(f"Bispectrum type {typ} has second order variable {typ[-1]} not from expected 'k' or 'w'.")
        L13_fac = product_func(L1, L3, L2)
        L23_fac = product_func(L2, L3, L1)
        res = 2 * L12_dot * ((L13_fac * M1) - (L23_fac * M2))/(L1**2 * L2**2)
        res[np.isnan(res)] = 0
        return res

    def _bispectrum_angle(self, typ, L1, L2, theta12, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold"):
        M1, M2, _ = self._bispectra_prep(typ, L1, L2, None, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins, gal_distro=gal_distro)
        return np.sin(2 * theta12) * (M1 - M2)

    def _get_lens_delta_bis(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro="LSST_gold"):
        if "w" in typ or typ == "kkk" or "k" not in typ:
            return 0
        if M_spline:
            self._build_M_splines_lens_delta(typ, nu, gal_bins, gal_distro=gal_distro)
        if "kk" in ''.join(sorted(typ)):
            if typ[0] == "k":  # kak
                print(f"Including lensed delta {'kk'+typ[1]} bispectrum")
                return self._lens_delta_bispectrum("kk"+typ[1], L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            else:  # akk
                print(f"Including lensed delta {'kk' + typ[0]} bispectrum")
                return self._lens_delta_bispectrum("kk"+typ[0], L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
        print(f"Including lensed delta {'k' + typ[1] + typ[0]} and {typ[0] + 'k' + typ[1]} bispectra")
        lens_bi1 = self._lens_delta_bispectrum("k" + typ[1] + typ[0], L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
        lens_bi2 = self._lens_delta_bispectrum(typ[0] + "k" + typ[1], L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
        return lens_bi1 + lens_bi2


    def get_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), gal_distro="LSST_gold", lens_delta=False):
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
        self._check_type(typ)
        if M_spline:
            self._build_M_splines(typ, nu, gal_bins, gal_distro=gal_distro)
        if L3 is not None:
            if typ == "kkk":
                b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline, zmin, zmax)
                return b1 + b2 + b3
            b = self._bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            if "kk" in ''.join(sorted(typ)) and "w" not in typ:
                if typ[0] == "k":   #kak
                    b += self._bispectrum(typ, L3, L2, L1, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
                else:   #akk
                    b += self._bispectrum(typ, L1, L3, L2, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
            if lens_delta:
                print("Including lensed delta bispectra terms")
                b += self._get_lens_delta_bis(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins, gal_distro)
            return b
        return self._bispectrum_angle(typ, L1, L2, theta, M_spline, zmin, zmax, nu, gal_bins, gal_distro=gal_distro)
