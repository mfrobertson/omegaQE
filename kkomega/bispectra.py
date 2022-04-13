from modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum to leading order in the Post Born approximation.

    """

    def __init__(self):
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
        self._mode = Modecoupling()
        self._M_kk_spline = None
        self._M_gk_spline = None
        self._M_kg_spline = None
        self._M_gg_spline = None
        self._M_Ik_spline = None
        self._M_kI_spline = None
        self._M_II_spline = None
        self._M_gI_spline = None
        self._M_Ig_spline = None
        #if M_spline:
            #self.build_M_spline(ells_sample, M_matrix, zmin, zmax)

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _bispectra_prep(self, typ,  L1, L2, L3=None, M_spline=False, zmin=0, zmax=None, nu=857e9):
        L12_dot = None
        if L3 is not None:
            L12_dot = self._triangle_dot_product(L1, L2, L3)
        if typ == "kappa-kappa-kappa" or typ == "kkk" or typ == "kappa-kappa-omega" or typ == "kkw":
            sign = 1
            if M_spline:
                M1 = self._M_kk_spline.ev(L1, L2)
                M2 = self._M_kk_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="kappa-kappa", zmin=zmin, zmax=zmax)
            M2 = self._mode.components(L2, L1, typ="kappa-kappa", zmin=zmin, zmax=zmax)
            return M1, M2, L12_dot, sign
        if typ == "gal-gal-omega" or typ == "ggw":
            sign = 1
            if M_spline:
                M1 = self._M_gg_spline.ev(L1, L2)
                M2 = self._M_gg_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="gal-gal", zmin=zmin, zmax=zmax)
            M2 = self._mode.components(L2, L1, typ="gal-gal", zmin=zmin, zmax=zmax)
            return M1, M2, L12_dot, sign
        if typ == "gal-kappa-omega" or typ == "gkw":
            sign = -1
            if M_spline:
                M1 = self._M_gk_spline.ev(L1, L2)
                M2 = self._M_kg_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="gal-kappa", zmin=zmin, zmax=zmax)
            M2 = self._mode.components(L2, L1, typ="kappa-gal", zmin=zmin, zmax=zmax)
            return M1, M2, L12_dot, sign
        if typ == "kappa-gal-omega" or typ == "kgw":
            sign = -1
            if M_spline:
                M1 = self._M_kg_spline.ev(L1, L2)
                M2 = self._M_gk_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="kappa-gal", zmin=zmin, zmax=zmax)
            M2 = self._mode.components(L2, L1, typ="gal-kappa", zmin=zmin, zmax=zmax)
            return M1, M2, L12_dot, sign
        if typ == "cib-cib-omega" or typ == "IIw":
            sign = 1
            if M_spline:
                M1 = self._M_II_spline.ev(L1, L2)
                M2 = self._M_II_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="cib-cib", zmin=zmin, zmax=zmax, nu=nu)
            M2 = self._mode.components(L2, L1, typ="cib-cib", zmin=zmin, zmax=zmax, nu=nu)
            return M1, M2, L12_dot, sign
        if typ == "cib-kappa-omega" or typ == "Ikw":
            sign = -1
            if M_spline:
                M1 = self._M_Ik_spline.ev(L1, L2)
                M2 = self._M_kI_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="cib-kappa", zmin=zmin, zmax=zmax, nu=nu)
            M2 = self._mode.components(L2, L1, typ="kappa-cib", zmin=zmin, zmax=zmax, nu=nu)
            return M1, M2, L12_dot, sign
        if typ == "kappa-cib-omega" or typ == "kIw":
            sign = -1
            if M_spline:
                M1 = self._M_kI_spline.ev(L1, L2)
                M2 = self._M_Ik_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="kappa-cib", zmin=zmin, zmax=zmax, nu=nu)
            M2 = self._mode.components(L2, L1, typ="cib-kappa", zmin=zmin, zmax=zmax, nu=nu)
            return M1, M2, L12_dot, sign
        if typ == "gal-cib-omega" or typ == "gIw":
            sign = -1
            if M_spline:
                M1 = self._M_gI_spline.ev(L1, L2)
                M2 = self._M_Ig_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="gal-cib", zmin=zmin, zmax=zmax, nu=nu)
            M2 = self._mode.components(L2, L1, typ="cib-gal", zmin=zmin, zmax=zmax, nu=nu)
            return M1, M2, L12_dot, sign
        if typ == "cib-gal-omega" or typ == "Igw":
            sign = -1
            if M_spline:
                M1 = self._M_Ig_spline.ev(L1, L2)
                M2 = self._M_gI_spline.ev(L2, L1)
                return M1, M2, L12_dot, sign
            M1 = self._mode.components(L1, L2, typ="cib-gal", zmin=zmin, zmax=zmax, nu=nu)
            M2 = self._mode.components(L2, L1, typ="gal-cib", zmin=zmin, zmax=zmax, nu=nu)
            return M1, M2, L12_dot, sign

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot, sign = self._bispectra_prep("kappa-kappa-kappa", L1, L2, L3, M_spline, zmin, zmax)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return sign * 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L3, L1, L2, M_spline, zmin, zmax)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L2, L3, L1, M_spline, zmin, zmax)

    def _build_M_spline(self, typ, ells_sample, M_matrix, zmin, zmax, nu):
        if ells_sample is not None and M_matrix is not None:
            return self._mode.spline(ells_sample, M_matrix, typ=typ)
        if ells_sample is None:
            return self._mode.spline(typ=typ, zmin=zmin, zmax=zmax, nu=nu)
        return self._mode.spline(ells_sample, typ=typ, zmin=zmin, zmax=zmax, nu=nu)

    def build_M_spline(self, typ="kappa-kappa", ells_sample=None, M_matrix=None, zmin=0, zmax=None, nu=857e9):
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
        M_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax, nu=nu)
        if typ == "kappa-kappa":
            self._M_kk_spline = M_spline
            return
        if typ == "gal-kappa":
            self._M_gk_spline = M_spline
            return
        if typ == "kappa-gal":
            self._M_kg_spline = M_spline
            return
        if typ == "gal-gal":
            self._M_gg_spline = M_spline
            return
        if typ == "cib-kappa":
            self._M_Ik_spline = M_spline
            return
        if typ == "kappa-cib":
            self._M_kI_spline = M_spline
            return
        if typ == "cib-cib":
            self._M_II_spline = M_spline
            return
        if typ == "cib-gal":
            self._M_Ig_spline = M_spline
            return
        if typ == "gal-cib":
            self._M_gI_spline = M_spline
            return

    def _build_M_splines(self, typ, nu):
        self._check_type(typ)
        if typ == "kappa-kappa-kappa" or typ == "kkk" or typ == "kappa-kapppa-omega" or typ == "kkw":
            if self._M_kk_spline is None:
                self.build_M_spline(typ="kappa-kappa")
            return
        if typ == "kappa-gal-kappa" or typ == "kgw" or typ == "gal-kapppa-omega" or typ == "gkw":
            if self._M_kg_spline is None:
                self.build_M_spline(typ="kappa-gal")
            if self._M_gk_spline is None:
                self.build_M_spline(typ="gal-kappa")
            return
        if typ == "gal-gal-kappa" or typ == "ggw":
            if self._M_gg_spline is None:
                self.build_M_spline(typ="gal-gal")
            return
        if typ == "cib-cib-kappa" or typ == "IIw":
            if self._M_II_spline is None:
                self.build_M_spline(typ="cib-cib")
            return
        if typ == "kappa-cib-kappa" or typ == "kIw" or typ == "cib-kapppa-omega" or typ == "Ikw":
            if self._M_kI_spline is None:
                self.build_M_spline(typ="kappa-cib")
            if self._M_Ik_spline is None:
                self.build_M_spline(typ="cib-kappa")
            return
        if typ == "gal-cib-kappa" or typ == "gIw" or typ == "cib-gal-omega" or typ == "Igw":
            if self._M_gI_spline is None:
                self.build_M_spline(typ="gal-cib")
            if self._M_Ig_spline is None:
                self.build_M_spline(typ="cib-gal")
            return
        raise UserWarning(f"Failed to build Mode Coupling splines for type {typ}")

    def _check_type(self, typ):
        typs = ["kappa-kappa-kappa", "kappa-kappa-omega", "kappa-gal-omega", "gal-kappa-omega", "gal-gal-omega",
                "cib-cib-omega", "cib-kappa-omega", "kappa-cib-omega", "gal-cib-omega", "cib-gal-omega", "kkk", "kkw",
                "kgw", "gkw", "ggw", "IIw", "Ikw", "kIw", "gIw", "Igw"]
        if typ not in typs:
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

    def _bispectrum(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu):
        M1, M2, L12_dot, sign = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return sign * 2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _bispectrum_angle(self, typ, L1, L2, theta, M_spline, zmin, zmax, nu):
        M1, M2, _, sign = self._bispectra_prep(typ, L1, L2, None, M_spline, zmin, zmax, nu=nu)
        return sign * np.sin(2 * theta) * (M1 - M2)

    def get_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None, nu=857e9):
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
            self._build_M_splines(typ=typ, nu=nu)
        if L3 is not None:
            if typ == "kappa-kappa-kappa" or typ == "kkk":
                b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline, zmin, zmax)
                return b1 + b2 + b3
            return self._bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu)
        return self._bispectrum_angle(typ, L1, L2, theta, M_spline, zmin, zmax, nu)
