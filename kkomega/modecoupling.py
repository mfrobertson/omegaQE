from powerspectra import Powerspectra
from cosmology import Cosmology
from maths import Maths
import numpy as np


class Modecoupling:
    """
    Calculates the mode-coupling matrix.

    Attributes
    ----------
    weyl_PK : object
        CAMB RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self):
        """
        Constructor

        """
        self._cosmo = Cosmology()
        self._powerspectra = Powerspectra()
        self._maths = Maths()
        self.weyl_PK = self._powerspectra.weyl_PK

    def _vectorise_ells(self, ells, ndim):
        if np.size(ells) == 1:
            return ells
        if ndim == 1:
            return ells[:, None]
        if ndim == 2:
            return ells[:, None, None]

    def _vectorise_zs(self, zs, Nells):
        ndim = zs.ndim
        if ndim == 1:
            return np.repeat(zs[np.newaxis, :], Nells, 0)
        if ndim == 2:
            return np.repeat(zs[np.newaxis, :, :], Nells, 0)

    def _integral_prep(self, Nchi, zmin, zmax):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        if zmax is not None:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        else:
            Chi_max = self._cosmo.get_chi_star()
        Chis = np.linspace(Chi_min, Chi_max, Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        window = self._cosmo.window(Chis, self._cosmo.get_chi_star())
        return zs, Chis, dChi, window

    def _components(self, ells1, ells2, star, Nchi, kmin, kmax, zmin, zmax, extended, recalc_weyl):
        zs, Chis, dChi, win = self._integral_prep(Nchi, zmin, zmax)
        Nells1 = np.size(ells1)
        ells1_vec = self._vectorise_ells(ells1, zs.ndim)
        zs = self._vectorise_zs(zs, Nells1)
        if star:
            Chi_source2 = self._cosmo.get_chi_star()
        else:
            Chi_source2 = None
        Cl_kappa = self._powerspectra.get_kappa_ps_2source(ells2, Chis, Chi_source2,
                                                           recalc_weyl=recalc_weyl)
        if recalc_weyl:
            self.weyl_PK = self._powerspectra.weyl_PK
        if extended:
            ks = (ells1_vec + 0.5) / Chis
        else:
            ks = ells1_vec / Chis
        step = self._maths.rectangular_pulse_steps(ks, kmin, kmax)
        weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=False, scaled=False)
        I = step * weyl_ps / Chis ** 2 * dChi * win ** 2 * Cl_kappa
        if extended:
            return I.sum(axis=1) * (ells1 + 0.5) ** 4
        return I.sum(axis=1) * ells1 ** 4

    def _matrix(self, ells1, ells2, star, Nchi, kmin, kmax, zmin, zmax, extended, recalc_weyl):
        M = np.ones((np.size(ells1), np.size(ells2)))
        for iii, ell1 in enumerate(ells1):
            M[iii, :] = self._components(np.ones(np.size(ells2))*ell1, ells2, star, Nchi, kmin, kmax, zmin, zmax, extended, recalc_weyl)
        return M

    def components(self, ells1, ells2, star=True, Nchi=100, kmin=0, kmax=100, zmin=0, zmax=None, extended=True, recalc_weyl=False):
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
        zmin :

        zmax :

        extended : bool
            Use extended Limber approximation.
        recalc_weyl : bool
            Recalculate the Weyl potential power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        ndarray
            1D array of the matrix components at [ells, ells2].
        """
        return self._components(ells1, ells2, star, Nchi, kmin, kmax, zmin, zmax, extended, recalc_weyl)

    def spline(self, ells_sample=None, M_matrix=None, star=True, Nchi=100, kmin=0, kmax=100, zmin=0, zmax=None, extended=True, recalc_weyl=False):
        """
        Produces 2D spline of the mode coupling matrix.

        Parameters
        ----------
        ells_sample : ndarray
            1D array of sample multipole moments. If not M_matrix is supplied these will be used for generating the spline.
        M_matrix : ndarray
            2D array of the modecoupling matrix at calculated at the positions given by ells_sample.
        Nchi : int
            The number of steps in the integral during the calculation.
        kmin : int or float
            Minimum wavelength to be used in the calculation.
        kmax : int or float
            Minimum wavelength to be used in the calculation.
        zmin :

        zmax :

        extended : bool
            Use extended Limber approximation.
        recalc_weyl : bool
            Recalculate the Weyl potential power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        RectBivariateSpline
            Returns the resulting spline object of the mode coupling matrix. RectBivariateSpline(ells1,ells2) will produce matrix. RectBivariateSpline.ev(ells1,ells2) will calculate components of the matrix.
        """
        from scipy.interpolate import RectBivariateSpline
        if ells_sample is not None and M_matrix is not None:
            return RectBivariateSpline(ells_sample, ells_sample, M_matrix)
        if ells_sample is None:
            ells_sample = self.generate_sample_ells()
        M = self._matrix(ells_sample, ells_sample, star, Nchi, kmin, kmax, zmin, zmax, extended, recalc_weyl)
        return RectBivariateSpline(ells_sample, ells_sample, M)

    def generate_sample_ells(self, ellmax=10000, Nells=100):
        """
        Produces optimised sample of multipole moments for input into spline build.

        Parameters
        ----------
        ellmax : int or float
            The maximum multipole moment to be sampled during generation of the interpolator.
        Nells : int
            Number of sample multipole moments along one dimension. Total number of samples for the matrix will be Nells * Nells.

        Returns
        -------
        ndarray
            1D array of multipole momnets to be used as input for spline build.
        """
        if ellmax <= 400:
            return np.linspace(1, 400, Nells)
        if ellmax <= 1600:
            ells_400 = np.linspace(1, 400, Nells // 2)
            Nells_remaining = Nells - np.size(ells_400)
            ells_remaining = np.linspace(401, ellmax, Nells_remaining)
            return np.concatenate((ells_400, ells_remaining))
        ells_400 = np.linspace(1, 400, Nells//3)
        ells_1600 = np.linspace(401, 1600, Nells//3)
        Nells_remaining = Nells - np.size(ells_400) - np.size(ells_1600)
        ells_remaining = np.linspace(1601, ellmax, Nells_remaining)
        return np.concatenate((ells_400, ells_1600, ells_remaining))
