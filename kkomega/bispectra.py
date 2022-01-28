from modecoupling import Modecoupling
import numpy as np


class Bispectra:
    """
    Calculates the convergence and convergence-rotation bispectrum.

    """

    def __init__(self): pass

    def _triangle_dot_product(self, mag1, mag2, mag3):
        return (mag1**2 + mag2**2 - mag3**2)/2

    def _triangle_cross_product(self, mag1, mag2, mag3):
        s = (mag1 + mag2 + mag3)/2
        return 2 * np.sqrt(s*(s-mag1)*(s-mag2)*(s-mag3))

    def _bispectra_prep(self, L1, L2, L3):
        mode = Modecoupling()
        M1 = mode.components(L1, L2)
        M2 = mode.components(L2, L1)
        L12_dot = self._triangle_dot_product(L1, L2, L3)
        return M1, M2, L12_dot

    def _kappa_kappa_kappa(self, L1, L2, L3):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3)
        L13_dot = self._triangle_dot_product(L1, L3, L2)
        L23_dot = self._triangle_dot_product(L2, L3, L1)
        return 2*L12_dot*(L13_dot*M1 + L23_dot*M2)/(L1**2 * L2**2)

    def _kappa_kappa_omega(self, L1, L2, L3):
        M1, M2, L12_dot = self._bispectra_prep(L1, L2, L3)
        L13_cross = self._triangle_cross_product(L1, L3, L2)
        L23_cross = self._triangle_cross_product(L2, L3, L1)
        return 2*L12_dot*(L13_cross*M1 + L23_cross*M2)/(L1**2 * L2**2)

    def get_convergence_bispectrum(self, L1, L2, L3):
        """
        Calculates the convergence bispectrum 'kappa kappa kappa' for one leading order post-Born convergence term.

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
        return self._kappa_kappa_kappa(L1, L2, L3)

    def get_convergence_rotation_bispectrum(self, L1, L2, L3):
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
        return self._kappa_kappa_omega(L1, L2, L3)
