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
        #if M_spline:
            #self.build_M_spline(ells_sample, M_matrix, zmin, zmax)

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _bispectra_prep(self, typ,  L1, L2, L3=None, M_spline=False, zmin=0, zmax=None):
        if M_spline:
            if typ == "kappa-kappa":
                M1 = self._M_kk_spline.ev(L1, L2)
                M2 = self._M_kk_spline.ev(L2, L1)
            elif typ == "gal-gal":
                M1 = self._M_gg_spline.ev(L1, L2)
                M2 = self._M_gg_spline.ev(L2, L1)
            elif typ == "kappa-gal" or typ == "gal-kappa":
                M1 = self._M_gk_spline.ev(L1, L2)
                M2 = self._M_kg_spline.ev(L2, L1)
            elif typ == "cib-cib":
                M1 = self._M_II_spline.ev(L1, L2)
                M2 = self._M_II_spline.ev(L2, L1)
            elif typ == "kappa-cib" or typ == "cib-kappa":
                M1 = self._M_Ik_spline.ev(L1, L2)
                M2 = self._M_kI_spline.ev(L2, L1)
        else:
            if typ == "kappa-gal" or typ == "gal-kappa":
                M1 = self._mode.components(L1, L2, typ="gal-kappa", zmin=zmin, zmax=zmax)
                M2 = self._mode.components(L2, L1, typ="kappa-gal", zmin=zmin, zmax=zmax)
            elif typ == "kappa-cib" or typ == "cib-kappa":
                M1 = self._mode.components(L1, L2, typ="cib-kappa", zmin=zmin, zmax=zmax)
                M2 = self._mode.components(L2, L1, typ="kappa-cib", zmin=zmin, zmax=zmax)
            else:
                M1 = self._mode.components(L1, L2, typ=typ, zmin=zmin, zmax=zmax)
                M2 = self._mode.components(L2, L1, typ=typ, zmin=zmin, zmax=zmax)
        if L3 is not None:
            L12_dot = self._triangle_dot_product(L1, L2, L3)
            return M1, M2, L12_dot
        return M1, M2

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("kappa-kappa", L1, L2, L3, M_spline, zmin, zmax)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L3, L1, L2, M_spline, zmin, zmax)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L2, L3, L1, M_spline, zmin, zmax)

    def _kappa1_kappa1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("kappa-kappa", L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2*L12_dot*(L13_cross*M1 - L23_cross*M2)/(L1**2 * L2**2)

    def _kappa1_kappa1_omega2_angle(self, L1, L2, theta, M_spline, zmin, zmax):
        M1, M2 = self._bispectra_prep("kappa-kappa", L1, L2, None, M_spline, zmin, zmax)
        return np.sin(2 * theta) * (M1 - M2)

    def _gal1_gal1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("gal-gal", L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _gal1_gal1_omega2_angle(self, L1, L2, theta, M_spline, zmin, zmax):
        M1, M2 = self._bispectra_prep("gal-gal", L1, L2, None, M_spline, zmin, zmax)
        return np.sin(2 * theta) * (M1 - M2)

    def _gal1_kappa1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("gal-kappa", L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return -2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _gal1_kappa1_omega2_angle(self, L1, L2, theta, M_spline, zmin, zmax):
        M1, M2 = self._bispectra_prep("gal-kappa", L1, L2, None, M_spline, zmin, zmax)
        return -np.sin(2 * theta) * (M1 - M2)

    def _cib1_cib1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("cib-cib", L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _cib1_cib1_omega2_angle(self, L1, L2, theta, M_spline, zmin, zmax):
        M1, M2 = self._bispectra_prep("cib-cib", L1, L2, None, M_spline, zmin, zmax)
        return np.sin(2 * theta) * (M1 - M2)

    def _cib1_kappa1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep("cib-kappa", L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return -2 * L12_dot * (L13_cross * M1 - L23_cross * M2)/(L1**2 * L2**2)

    def _cib1_kappa1_omega2_angle(self, L1, L2, theta, M_spline, zmin, zmax):
        M1, M2 = self._bispectra_prep("cib-kappa", L1, L2, None, M_spline, zmin, zmax)
        return -np.sin(2 * theta) * (M1 - M2)


    def _build_M_spline(self, typ, ells_sample, M_matrix, zmin, zmax):
        if ells_sample is not None and M_matrix is not None:
            return self._mode.spline(ells_sample, M_matrix, typ=typ)
        if ells_sample is None:
            return self._mode.spline(typ=typ, zmin=zmin, zmax=zmax)
        return self._mode.spline(ells_sample, typ=typ, zmin=zmin, zmax=zmax)

    def build_M_spline(self, typ="kappa-kappa", ells_sample=None, M_matrix=None, zmin=0, zmax=None):
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
        if typ == "kappa-kappa":
            self._M_kk_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "gal-kappa":
            self._M_gk_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "kappa-gal":
            self._M_kg_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "gal-gal":
            self._M_gg_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "cib-kappa":
            self._M_Ik_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "kappa-cib":
            self._M_kI_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return
        if typ == "cib-cib":
            self._M_II_spline = self._build_M_spline(typ, ells_sample, M_matrix, zmin, zmax)
            return

    def get_convergence_bispectrum(self, L1, L2, L3, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the convergence bispectrum 'kappa kappa kappa' to leading order in the Post Born correction.

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
        if M_spline and self._M_kk_spline is None:
            self.build_M_spline(typ="kappa-kappa")
        b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline, zmin, zmax)
        b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline, zmin, zmax)
        b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline, zmin, zmax)
        return b1 + b2 + b3

    def get_convergence_rotation_bispectrum(self, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the convergence-rotation bispectrum 'kappa kappa omega' for the leading order rotation term.

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
        if M_spline and self._M_kk_spline is None:
            self.build_M_spline(typ="kappa-kappa")
        if L3 is not None:
            return self._kappa1_kappa1_omega2(L1, L2, L3, M_spline, zmin, zmax)
        return self._kappa1_kappa1_omega2_angle(L1, L2, theta, M_spline, zmin, zmax)

    def get_gal_rotation_bispectrum(self, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the galaxy-rotation bispectrum 'g g omega' for the leading order rotation term.

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
        if M_spline and self._M_gg_spline is None:
            self.build_M_spline(typ="gal-gal")
        if L3 is not None:
            return self._gal1_gal1_omega2(L1, L2, L3, M_spline, zmin, zmax)
        return self._gal1_gal1_omega2_angle(L1, L2, theta, M_spline, zmin, zmax)

    def get_gal_convergence_rotation_bispectrum(self, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the galaxy-convergence-rotation bispectrum 'g kappa omega' for the leading order rotation term.

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
        if M_spline and self._M_gk_spline is None:
            self.build_M_spline(typ="gal-kappa")
            self.build_M_spline(typ="kappa-gal")
        if L3 is not None:
            return self._gal1_kappa1_omega2(L1, L2, L3, M_spline, zmin, zmax)
        return self._gal1_kappa1_omega2_angle(L1, L2, theta, M_spline, zmin, zmax)

    def get_cib_rotation_bispectrum(self, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the galaxy-rotation bispectrum 'g g omega' for the leading order rotation term.

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
        if M_spline and self._M_II_spline is None:
            self.build_M_spline(typ="cib-cib")
        if L3 is not None:
            return self._cib1_cib1_omega2(L1, L2, L3, M_spline, zmin, zmax)
        return self._cib1_cib1_omega2_angle(L1, L2, theta, M_spline, zmin, zmax)

    def get_cib_convergence_rotation_bispectrum(self, L1, L2, L3=None, theta=None, M_spline=False, zmin=0, zmax=None):
        """
        Calculates the galaxy-convergence-rotation bispectrum 'g kappa omega' for the leading order rotation term.

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
        if M_spline and self._M_Ik_spline is None:
            self.build_M_spline(typ="cib-kappa")
            self.build_M_spline(typ="kappa-cib")
        if L3 is not None:
            return self._cib1_kappa1_omega2(L1, L2, L3, M_spline, zmin, zmax)
        return self._cib1_kappa1_omega2_angle(L1, L2, theta, M_spline, zmin, zmax)