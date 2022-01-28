from powerspectra import Powerspectra
from cosmology import Cosmology
import numpy as np

class Modecoupling:
    """
    Calculates the mode-coupling matrix.

    Attributes
    ----------
    weyl_PK : object
        RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self):
        """
        Constructor

        """
        self._cosmo = Cosmology()
        self._powerspectra = Powerspectra()
        self.weyl_PK = self._powerspectra.weyl_PK

    def _integral_prep(self, Nchi):
        Chi_max = self._cosmo.get_chi_star()
        Chis = np.linspace(0, Chi_max, Nchi)
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        zs = zs[1:]
        Chis = Chis[1:]
        window = self._cosmo.window(Chis, Chi_max)
        return zs, Chis, dChi, window

    def components(self, ells1, ells2, Nchi=100, kmin=0, kmax=100, extended=True, recalc_weyl=True):
        """
        Performs the calculation for extracting components of the mode-coupling matrix.

        Parameters
        ----------
        ells1 : int or float or ndarray
            The multipole moment(s). Essentially the first indices of the matrix.
        ells2 : int or float or ndarray
            The multipole moment(s) with same dimensions as ell1. Essentially the second indices of the matrix.
        Nchi : int
            The number of steps in the integral during the calculation.
        kmin : int or float
            Minimum wavelength to be used in the calculation.
        kmax : int or float
            Minimum wavelength to be used in the calculation.
        extended : bool
            Use extended Limber approximation.
        recalc_weyl : bool
            Recalculate the Weyl potential power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        ndarray
            The matrix components at (ells, ells2).
        """
        zs, Chis, dChi, win = self._integral_prep(Nchi)
        M = np.zeros(np.shape(ells1))
        Cl_kappa = self._powerspectra.get_kappa_ps_2source(ells2, Chis, self._cosmo.get_chi_star(), recalc_weyl=recalc_weyl)
        if recalc_weyl:
            self.weyl_PK = self._powerspectra.weyl_PK
        for iii, ell1 in enumerate(ells1):
            if extended:
                ks = (ell1 + 0.5)/ Chis
            else:
                ks = ell1 / Chis
            step = self._cosmo.heaviside(ks, kmin, kmax)
            weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=False, scaled=False)
            I = step * weyl_ps/Chis**2 * dChi * win ** 2 * Cl_kappa[iii]
            if extended:
                M[iii] = np.sum(I) * (ell1 + 0.5) ** 4
            else:
                M[iii] = np.sum(I) * ell1 ** 4
        return M
