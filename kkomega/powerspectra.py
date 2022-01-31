import numpy as np
from cosmology import Cosmology

class Powerspectra:
    """
    Calculates Limber approximated CMB lensing power spectra at the order of the Born approximation.

    Attributes
    ----------
    weyl_PK : object
        RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self):
        """
        Constructor.

        """
        self._cosmo = Cosmology()
        self.weyl_PK = self._get_weyl_PK()

    def _get_weyl_PK(self, ellmax=1000, Nchi=100):
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

    def _integral_prep(self, Nchi, zmin, zmax):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        if zmax is None:
            Chi_max = self._cosmo.get_chi_star()
        else:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        Chis = np.linspace(Chi_min, Chi_max, Nchi)
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        zs = zs[1:]
        Chis = Chis[1:]
        window = self._cosmo.window(Chis, Chi_max)
        return zs, Chis, dChi, window

    def _Cl_phi(self, ells, Nchi, zmin=0, zmax=None, kmin=0, kmax=100, extended=False):
        zs, Chis, dChi, win = self._integral_prep(Nchi, zmin, zmax)
        Cl_phi = np.zeros(np.shape(ells))
        for iii, ell in enumerate(ells):
            if extended:
                ks = (ell + 0.5)/ Chis
            else:
                ks = ell / Chis
            step = self._cosmo.heaviside(ks, kmin, kmax)
            weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=True, scaled=False)
            I = step * Chis * weyl_ps * dChi * win ** 2
            if extended:
                Cl_phi[iii] = np.sum(I) / (ell + 0.5) ** 3 * 8 * np.pi ** 2
            else:
                Cl_phi[iii] = np.sum(I) / ell ** 3 * 8 * np.pi ** 2
        return Cl_phi

    def _Cl_kappa(self, ells, Nchi, zmin=0, zmax=None, kmin=0, kmax=100, extended=False):
        zs, Chis, dChi, win = self._integral_prep(Nchi, zmin, zmax)
        Cl_kappa = np.zeros(np.shape(ells))
        for iii, ell in enumerate(ells):
            if extended:
                ks = (ell + 0.5) / Chis
            else:
                ks = ell / Chis
            step = self._cosmo.heaviside(ks, kmin, kmax)
            weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=False, scaled=False)
            I = step * weyl_ps/(Chis)**2 * dChi * win ** 2
            if extended:
                Cl_kappa[iii] = np.sum(I) * (ell + 0.5)** 4
            else:
                Cl_kappa[iii] = np.sum(I) * ell** 4
        return Cl_kappa

    def _Cl_kappa_2source(self, ells, Chi_source1, Chi_source2, Nchi, kmin=0, kmax=100, extended=False):
        Cl_kappa = np.zeros((np.size(ells), np.size(Chi_source1)))
        zmax = self._cosmo.Chi_to_z(Chi_source1)
        zs, Chis, dChi, _ = self._integral_prep(Nchi, zmin=0, zmax=zmax)
        win1 = self._cosmo.window(Chis, Chi_source1)
        win2 = self._cosmo.window(Chis, Chi_source2)
        for iii, ell in enumerate(ells):
            if extended:
                ks = (ell + 0.5) / Chis
            else:
                ks = ell / Chis
            step = self._cosmo.heaviside(ks, kmin, kmax)
            weyl_ps = self._cosmo.get_weyl_ps(self.weyl_PK, zs, ks, curly=False, scaled=False)
            I = step * weyl_ps / Chis ** 2 * dChi * win1 * win2
            if extended:
                Cl_kappa[iii] = I.sum(axis=0) * (ell + 0.5) ** 4
            else:
                Cl_kappa[iii] = I.sum(axis=0) * ell ** 4
        return Cl_kappa

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
        Return the Limber approximated lensing convergence power spectrum for two source planes.

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


if __name__ == "__main__":
    pass