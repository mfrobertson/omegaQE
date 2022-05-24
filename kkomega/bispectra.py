from modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum to leading order in the Post Born approximation.

    """

    class M_spline:


        def __init__(self, spline, nu, gal_bins):
            self.spline = spline
            self.nu = nu
            self.gal_bins = gal_bins

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
        self._M_splines = dict.fromkeys(self._mode.get_M_types())

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _get_sign(self, typ):
        if typ == "kk":
            return 1
        if typ[0] == "k" or typ[1] == "k":
            return -1
        return 1

    def _bispectra_prep(self, typ,  L1, L2, L3=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None)):
        L12_dot = None
        if L3 is not None:
            L12_dot = self._triangle_dot_product(L1, L2, L3)
        M_typ1 = typ[:2]
        M_typ2 = M_typ1[::-1]
        sign = self._get_sign(M_typ1)
        if M_spline:
            M1 = self._M_splines[M_typ1].spline.ev(L1, L2)
            M2 = self._M_splines[M_typ2].spline.ev(L2, L1)
            return M1, M2, L12_dot, sign
        M1 = self._mode.components(L1, L2, typ=M_typ1, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins)
        M2 = self._mode.components(L2, L1, typ=M_typ2, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins)
        return M1, M2, L12_dot, sign

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot, sign = self._bispectra_prep("kkk", L1, L2, L3, M_spline, zmin, zmax)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return sign * 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L3, L1, L2, M_spline, zmin, zmax)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L2, L3, L1, M_spline, zmin, zmax)

    def _build_M_spline(self, typ, ells_sample, M_matrix, zmin, zmax, nu, gal_bins):
        if ells_sample is not None and M_matrix is not None:
            spline = self._mode.spline(ells_sample, M_matrix, typ=typ)
            return self.M_spline(spline, nu, gal_bins)
        if ells_sample is None:
            spline = self._mode.spline(typ=typ, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins)
            return self.M_spline(spline, nu, gal_bins)
        spline = self._mode.spline(ells_sample, typ=typ, zmin=zmin, zmax=zmax, nu=nu, gal_bins=gal_bins)
        return self.M_spline(spline, nu, gal_bins)

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

    def _build_M_splines(self, typ, nu, gal_bins):
        M_typ1 = typ[:2]
        M_typ2 = M_typ1[::-1]
        if self._M_splines[M_typ1] is not None:
            if self._M_splines[M_typ1].nu != nu or self._M_splines[M_typ1].gal_bins != gal_bins:
                self._M_splines[M_typ1] = self._build_M_spline(M_typ1, None, None, 0, None, nu, gal_bins)
                if M_typ2 != M_typ1:
                    self._M_splines[M_typ2] = self._build_M_spline(M_typ2, None, None, 0, None, nu, gal_bins)
            return
        self._M_splines[M_typ1] = self._build_M_spline(M_typ1, None, None, 0, None, nu, gal_bins)
        if M_typ2 != M_typ1:
            self._M_splines[M_typ2] = self._build_M_spline(M_typ2, None, None, 0, None, nu, gal_bins)

    def _check_type(self, typ):
        typs = self._mode.get_M_types() + "w"
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

    def _bispectrum(self, typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins):
        M1, M2, L12_dot, sign = self._bispectra_prep(typ, L1, L2, L3, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return sign * 2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _bispectrum_angle(self, typ, L1, L2, theta, M_spline, zmin, zmax, nu, gal_bins):
        M1, M2, _, sign = self._bispectra_prep(typ, L1, L2, None, M_spline, zmin, zmax, nu=nu, gal_bins=gal_bins)
        return sign * np.sin(2 * theta) * (M1 - M2)

    def get_bispectrum(self, typ, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None)):
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
            self._build_M_splines(typ, nu, gal_bins)
        if L3 is not None:
            if typ == "kkk":
                b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline, zmin, zmax)
                b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline, zmin, zmax)
                return b1 + b2 + b3
            return self._bispectrum(typ, L1, L2, L3, M_spline, zmin, zmax, nu, gal_bins)
        return self._bispectrum_angle(typ, L1, L2, theta, M_spline, zmin, zmax, nu, gal_bins)
