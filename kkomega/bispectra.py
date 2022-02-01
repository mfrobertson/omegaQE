from modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum to leading order in the Post Born approximation.

    """

    def __init__(self, M_spline=False, ellmax=10000, Nell=100):
        self._mode = Modecoupling()
        if M_spline:
            self._spline = self._mode.spline(ellmax, Nell)

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _bispectra_prep(self, L1, L2, L3, M_spline):
        if M_spline:
            M1 = self._spline.ev(L1, L2)
            M2 = self._spline.ev(L2, L1)
        else:
            M1 = self._mode.components(L1, L2)
            M2 = self._mode.components(L2, L1)
        L12_dot = self._triangle_dot_product(L1, L2, L3)
        return M1, M2, L12_dot

    def _kappa1_kappa1_kappa2(self, L1, L2, L3, M_spline):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3, M_spline)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa1_kappa2_kappa1(self, L1, L2, L3, M_spline):
        return self._kappa1_kappa1_kappa2(L3, L1, L2, M_spline)

    def _kappa2_kappa1_kappa1(self, L1, L2, L3, M_spline):
        return self._kappa1_kappa1_kappa2(L2, L3, L1, M_spline)

    def _kappa1_kappa1_omega2(self, L1, L2, L3, M_spline):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3, M_spline)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2*L12_dot*(L13_cross*M1 + L23_cross*M2)/(L1**2 * L2**2)

    def get_convergence_bispectrum(self, L1, L2, L3, M_spline=False):
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


        Returns
        -------
        float or ndarray
            The bispectrum.
        """
        b1 = self._kappa2_kappa1_kappa1(L1, L2, L3, M_spline)
        b2 = self._kappa1_kappa2_kappa1(L1, L2, L3, M_spline)
        b3 = self._kappa1_kappa1_kappa2(L1, L2, L3, M_spline)
        return b1 + b2 + b3

    def get_convergence_rotation_bispectrum(self, L1, L2, L3, M_spline=False):
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


        Returns
        -------
        float or ndarray
            The bispectrum.
        """
        return self._kappa1_kappa1_omega2(L1, L2, L3, M_spline)
