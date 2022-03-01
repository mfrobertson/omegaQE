import numpy as np
from cosmology import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline

class Powerspectra:
    """
    Calculates Limber approximated CMB lensing power spectra at the order of the Born approximation.

    Attributes
    ----------
    weyl_PK : object
        CAMB RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self):
        """
        Constructor.

        """
        self._cosmo = Cosmology()
        self.weyl_PK = self._get_weyl_PK()

    def _get_weyl_PK(self, ellmax=None, Nchi=None):
        if ellmax is None and Nchi is None:
            return self._cosmo.get_weyl_PK()
        zbuffer = 100
        zmax = self._cosmo.eta_to_z(self._cosmo.get_eta0() - self._cosmo.get_chi_star()) + zbuffer
        kbuffer = 10
        kmax = ellmax * Nchi/self._cosmo.get_chi_star() + kbuffer
        return self._cosmo.get_weyl_PK(kmax, zmax)

    def recalculate_weyl(self, ellmax,  Nchi):
        """
        Force recalculation of Weyl power spectrum interpolator, self.weyl_PK. This recalculation is optimised for power spectra calculations where ellmax and Nchi will be supplied as input parameters.

        Parameters
        ----------
        ellmax : float or int
            The maximum moment the power spectra will be calculated over.
        Nchi : int
            The number of steps in the integral during the power spectra calculation.
        Returns
        -------
        None
        """
        zbuffer = 100
        zmax = self._cosmo.eta_to_z(self._cosmo.get_eta0() - self._cosmo.get_chi_star()) + zbuffer
        kbuffer = 10
        kmax = ellmax * Nchi / self._cosmo.get_chi_star() + kbuffer
        self.weyl_PK = self._cosmo.get_weyl_PK(kmax, zmax)

    def get_weyl_ps(self, z, k, curly=False, scaled=True):
        """
        Returns the Weyl power spectrum, calculated from the interpolator self.weyl_PK.

        Parameters
        ----------
        z : int or float or ndarray
            Redshift.
        k : int or float or ndarray
            [Mpc^-1].
        curly : bool
            Return dimensionless power spectrum.
        scaled : bool
            Accept default CAMB scaling of Weyl potential by k^2.

        Returns
        -------
        ndarray
            Weyl power spectrum calculated at specific points z and k.
        """
        return self._cosmo.get_weyl_ps(self.weyl_PK, z, k, curly, scaled)

    def _vectorise_ells(self, ells, ndim):
        if np.size(ells) == 1:
            return ells
        if ndim == 1:
            return ells[:, None]
        if ndim == 2:
            return ells[:, None, None]

    def _vectorise_zs(self, Chis, Nells):
        ndim = Chis.ndim
        if ndim == 1:
            zs = self._cosmo.Chi_to_z(Chis)
            return np.repeat(zs[np.newaxis, :], Nells, 0)
        if ndim == 2:
            zs = np.ones(np.shape(Chis))
            for row in range(np.shape(Chis)[0]):
                zs[row] = self._cosmo.Chi_to_z(Chis[row])
            return np.repeat(zs[np.newaxis, :, :], Nells, 0)

    def _integral_prep(self, ells, Nchi, zmin, zmax, kmin, kmax, extended, curly):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        if zmax is None:
            Chi_max = self._cosmo.get_chi_star()
        else:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        Chis = np.linspace(Chi_min, Chi_max, Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        window = self._cosmo.window(Chis, self._cosmo.get_chi_star())
        Nells = np.size(ells)
        ells = self._vectorise_ells(ells, Chis.ndim)
        zs = self._vectorise_zs(Chis, Nells)
        if extended:
            ks = (ells + 0.5)/Chis
        else:
            ks = ells/Chis
        step = self._cosmo.rectangular_pulse_steps(ks, kmin, kmax)
        weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=curly, scaled=False)
        return step, Chis, weyl_ps, dChi, window

    def _Cl_phi(self, ells, Nchi, zmin=0, zmax=None, kmin=0, kmax=100, extended=False):
        step, Chis, weyl_ps, dChi, win = self._integral_prep(ells, Nchi, zmin, zmax, kmin, kmax, extended, curly=True)
        I = step * Chis * weyl_ps * dChi * win ** 2
        if extended:
            return I.sum(axis=1) / (ells + 0.5) ** 3 * 8 * np.pi ** 2
        return I.sum(axis=1) / ells ** 3 * 8 * np.pi ** 2

    def _Cl_kappa(self, ells, Nchi, zmin=0, zmax=None, kmin=0, kmax=100, extended=False):
        step, Chis, weyl_ps, dChi, win = self._integral_prep(ells, Nchi, zmin, zmax, kmin, kmax, extended, curly=False)
        I = step * weyl_ps/(Chis)**2 * dChi * win ** 2
        if extended:
            return I.sum(axis=1) * (ells + 0.5)** 4
        return I.sum(axis=1) * ells** 4

    def _Cl_kappa_2source(self, ells, Chi_source1, Chi_source2, Nchi, kmin=0, kmax=100, extended=True):
        zmax = self._cosmo.Chi_to_z(Chi_source1)
        step, Chis, weyl_ps, dChi, _ = self._integral_prep(ells, Nchi, 0, zmax, kmin, kmax, extended, curly=False)
        win1 = self._cosmo.window(Chis, Chi_source1)
        win2 = self._cosmo.window(Chis, Chi_source2)
        I = step * weyl_ps / Chis ** 2 * dChi * win1 * win2
        if np.size(Chi_source1) > 1:
            ells = self._vectorise_ells(ells, 1)
        if extended:
            return I.sum(axis=1) * (ells + 0.5) ** 4
        return I.sum(axis=1) * ells ** 4

    def get_phi_ps(self, ells, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        """
        Return the Limber approximated lensing potential power spectrum.

        Parameters
        ----------
        ells : int or float or ndarray
            Multipole moment(s) over which to calculate the power spectrum.
        Nchi : int
            Number of steps in the integral over Chi.
        zmin : int or float
            Minimum redshift to be used in the calculation.
        zmax : int or float or None
            Maximum redshift to be used in the calculation. None value will default yo surface of last scattering.
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
            1D ndarray of the lensing potential power spectrum calculated at the supplied ell values.
        """
        if recalc_weyl:
            self.weyl_PK = self._get_weyl_PK(np.max(ells), Nchi)
        return self._Cl_phi(ells, Nchi, zmin, zmax, kmin, kmax, extended)

    def get_kappa_ps(self, ells, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        """
        Return the Limber approximated lensing convergence power spectrum.

        Parameters
        ----------
        ells : int or float or ndarray
            Multipole moment(s) over which to calculate the power spectrum.
        Nchi : int
            Number of steps in the integral over Chi.
        zmin : int or float
            Minimum redshift to be used in the calculation.
        zmax : int or float or None
            Maximum redshift to be used in the calculation. None value will default yo surface of last scattering.
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
            1D ndarray of the lensing convergence power spectrum calculated at the supplied ell values.
        """
        if recalc_weyl:
            self.weyl_PK = self._get_weyl_PK(np.max(ells), Nchi)
        return self._Cl_kappa(ells, Nchi, zmin, zmax, kmin, kmax, extended)

    def get_kappa_ps_2source(self, ells, Chi_source1, Chi_source2, Nchi=100, kmin=0, kmax=100, extended=True, recalc_weyl=False):
        """
        Returns the Limber approximated lensing convergence power spectrum for two source planes.

        Parameters
        ----------
        ells : int or float or ndarray
            Multipole moments at which to calculate the power spectrum.
        Chi_source1 : int or float or ndarray
            Comoving radial distance(s) of the first source plane(s) [Mpc]. Will be used for the first window function, and as the integral limit.
        Chi_source2 : int or float
            Comoving radial distance of the second source plane [Mpc]. Will be used for the second window function.
        Nchi : int
            Number of steps in the integral over Chi.
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
            2D ndarray of the lensing convergence power spectrum calculated at the supplied ell and Chi_source1 values. Indexed by [ell, Chi_source1].         """
        if recalc_weyl:
            self.weyl_PK = self._get_weyl_PK(np.max(ells), Nchi)
        return self._Cl_kappa_2source(ells, Chi_source1, Chi_source2, Nchi, kmin, kmax, extended)

    def get_camb_omega_ps(self, ellmax=10000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return self._cosmo.get_omega_ps(2*ellmax)

    def get_ps_variance(self, ells, Cl, N0, auto=True):
        """

        Parameters
        ----------
        ells
        Cl
        N0
        auto

        Returns
        -------

        """
        if auto:
            return 2 / (2 * ells + 1) * (Cl + N0) ** 2
        return 2 / (2 * ells + 1) * (Cl**2 + (N0 * Cl))


if __name__ == "__main__":
    pass