from modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum to leading order in the Post Born approximation.

    """

    def __init__(self, M_spline=False, ells_sample=None, M_matrix=None, zmin=0, zmax=None):
        """
        Constructor.

        Parameters
        ----------
        M_spline : bool
            On instantiation, build a spline for estimation of the mode-coupling components whcih can be used for quicker calculation.
        ells_sample : ndarray
            1D array of sample multipole moments. If not M_matrix is supplied these will be used for generating the spline.
        M_matrix : ndarray
            2D array of the modecoupling matrix at calculated at the positions given by ells_sample.
        """
        self._mode = Modecoupling()
        if M_spline:
            self.build_M_spline(ells_sample, M_matrix, zmin, zmax)

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _bispectra_prep(self, L1, L2, L3, M_spline, zmin, zmax):
        if M_spline:
            M1 = self._spline.ev(L1, L2)
            M2 = self._spline.ev(L2, L1)
        else:
            M1 = self._mode.components(L1, L2, zmin=zmin, zmax=zmax)
            M2 = self._mode.components(L2, L1, zmin=zmin, zmax=zmax)
        L12_dot = self._triangle_dot_product(L1, L2, L3)
        return M1, M2, L12_dot

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3, M_spline, zmin, zmax)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L3, L1, L2, M_spline, zmin, zmax)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline, zmin, zmax):
        return self._kappa1_kappa1_kappa2(L2, L3, L1, M_spline, zmin, zmax)

    def _kappa1_kappa1_omega2(self, L1, L2, L3, M_spline, zmin, zmax):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3, M_spline, zmin, zmax)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2*L12_dot*(L13_cross*M1 - L23_cross*M2)/(L1**2 * L2**2)

    def build_M_spline(self, ells_sample=None, M_matrix=None, zmin=0, zmax=None):
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
        if ells_sample is not None and M_matrix is not None:
            self._spline = self._mode.spline(ells_sample, M_matrix)
            return
        if ells_sample is None:
            self._spline = self._mode.spline(zmin=zmin, zmax=zmax)
            return
        self._spline = self._mode.spline(ells_sample, zmin=zmin, zmax=zmax)

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
        if M_spline and self._spline is None:
            self._spline = self.build_M_spline()
        b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline, zmin, zmax)
        b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline, zmin, zmax)
        b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline, zmin, zmax)
        return b1 + b2 + b3

    def get_convergence_rotation_bispectrum(self, L1, L2, L3, M_spline=False, zmin=0, zmax=None):
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
        if M_spline and self._spline is None:
            self._spline = self.build_M_spline()
        return self._kappa1_kappa1_omega2(L1, L2, L3, M_spline, zmin, zmax)
