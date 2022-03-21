import numpy as np
from cosmology import Cosmology
from maths import Maths

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
        self._maths = Maths()
        self.weyl_PK = None
        self.matter_weyl_PK = None
        self.matter_PK = None

    def _get_PK(self, typ, ellmax=None, Nchi=None):
        if ellmax is None and Nchi is None:
            return self._cosmo.get_matter_PK(typ=typ)
        zbuffer = 100
        zmax = self._cosmo.eta_to_z(self._cosmo.get_eta0() - self._cosmo.get_chi_star()) + zbuffer
        kbuffer = 10
        kmax = ellmax * Nchi/self._cosmo.get_chi_star() + kbuffer
        return self._cosmo.get_matter_PK(kmax, zmax, typ=typ)

    def recalculate_PK(self, typ, ellmax,  Nchi):
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
        PK = self._get_PK(typ, ellmax, Nchi)
        if typ.lower() == "weyl":
            self.weyl_PK = PK
        elif typ.lower() == "matter-weyl" or typ.lower() == "weyl-matter":
            self.matter_weyl_PK = PK
        elif typ.lower() == "matter":
            self.matter_PK = PK

    def get_matter_ps(self, typ, z, k, curly=False, weyl_scaled=True):
        """
        Returns the matter power spectrum, calculated from the interpolator self.matter_PK.

        Parameters
        ----------
        z : int or float or ndarray
            Redshift.
        k : int or float or ndarray
            [Mpc^-1].
        curly : bool
            Return dimensionless power spectrum.

        Returns
        -------
        ndarray
            Matter power spectrum calculated at specific points z and k.
        """
        if typ.lower() == "weyl":
            if self.weyl_PK is None:
                self.weyl_PK = self._get_PK("weyl")
            return self._cosmo.get_matter_ps(self.weyl_PK, z, k, curly, weyl_scaled)
        elif typ.lower() == "matter-weyl" or typ.lower() == "weyl-matter":
            if self.matter_weyl_PK is None:
                self.matter_weyl_PK = self._get_PK("matter-weyl")
            return self._cosmo.get_matter_ps(self.matter_weyl_PK, z, k, curly, weyl_scaled, typ="matter-weyl")
        elif typ.lower() == "matter":
            if self.matter_PK is None:
                self.matter_PK = self._get_PK("matter")
            return self._cosmo.get_matter_ps(self.matter_PK, z, k, curly)

    def _get_matter_ps(self, typ, zs, ks, curly, weyl_scaled):
        return self.get_matter_ps(typ, zs, ks, curly, weyl_scaled)

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

    def _integral_prep(self, ells, Nchi, zmin, zmax, kmin, kmax, extended, curly, matter_ps_typ="weyl"):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        if zmax is None:
            Chi_max = self._cosmo.get_chi_star()
        else:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        Chis = np.linspace(Chi_min, Chi_max, Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        Nells = np.size(ells)
        ells = self._vectorise_ells(ells, Chis.ndim)
        zs = self._vectorise_zs(Chis, Nells)
        if extended:
            ks = (ells + 0.5)/Chis
        else:
            ks = ells/Chis
        step = self._maths.rectangular_pulse_steps(ks, kmin, kmax)
        matter_ps = self._get_matter_ps(matter_ps_typ, zs, ks, curly, weyl_scaled=False)
        return step, Chis, matter_ps, dChi

    def _Cl_phi(self, ells, Nchi, zmin, zmax, kmin, kmax, extended):
        step, Chis, weyl_ps, dChi = self._integral_prep(ells, Nchi, zmin, zmax, kmin, kmax, extended, curly=True)
        window = self._cosmo.cmb_lens_window(Chis, self._cosmo.get_chi_star())
        I = step * Chis * weyl_ps * dChi * window ** 2
        if extended:
            return I.sum(axis=1) / (ells + 0.5) ** 3 * 8 * np.pi ** 2
        return I.sum(axis=1) / ells ** 3 * 8 * np.pi ** 2

    def _Cl_kappa(self, ells, Nchi, zmin, zmax, kmin, kmax, extended):
        step, Chis, weyl_ps, dChi = self._integral_prep(ells, Nchi, zmin, zmax, kmin, kmax, extended, curly=False)
        window = self._cosmo.cmb_lens_window(Chis, self._cosmo.get_chi_star())
        I = step * weyl_ps/(Chis)**2 * dChi * window ** 2
        if extended:
            return I.sum(axis=1) * (ells + 0.5)** 4
        return I.sum(axis=1) * ells** 4

    def _Cl_kappa_2source(self, ells, Chi_source1, Chi_source2, Nchi, kmin, kmax, extended):
        zmax = self._cosmo.Chi_to_z(Chi_source1)
        step, Chis, weyl_ps, dChi= self._integral_prep(ells, Nchi, 0, zmax, kmin, kmax, extended, curly=False)
        window1 = self._cosmo.cmb_lens_window(Chis, Chi_source1)
        if Chi_source2 is not None:
            window2 = self._cosmo.cmb_lens_window(Chis, Chi_source2)
        else:
            window2 = window1
        I = step * weyl_ps / Chis ** 2 * dChi * window1 * window2
        if np.size(Chi_source1) > 1:
            ells = self._vectorise_ells(ells, 1)
        if extended:
            return I.sum(axis=1) * (ells + 0.5) ** 4
        return I.sum(axis=1) * ells ** 4

    def _Cl_gal_kappa(self, ells, Chi_source1, Nchi, kmin, kmax, extended):
        if Chi_source1 is None:
            Chi_source1 = self._cosmo.get_chi_star()
        zmax = self._cosmo.Chi_to_z(Chi_source1)
        step, Chis, matter_weyl_ps, dChi = self._integral_prep(ells, Nchi, 0, zmax, kmin, kmax, extended, curly=False, matter_ps_typ="matter-weyl")
        window1 = self._cosmo.cmb_lens_window(Chis, Chi_source1)
        window2 = self._cosmo.gal_cluster_window_Chi(Chis)
        I = step * matter_weyl_ps / (Chis ** 2) * dChi * window1 * window2
        if np.size(Chi_source1) > 1:
            ells = self._vectorise_ells(ells, 1)
        if extended:
            return I.sum(axis=1) * (ells + 0.5) ** 2
        return I.sum(axis=1) * ells ** 2

    def _Cl_gal(self, ells, Nchi, zmin, zmax, kmin, kmax, extended):
        step, Chis, matter_ps, dChi= self._integral_prep(ells, Nchi, zmin, zmax, kmin, kmax, extended, curly=False, matter_ps_typ="matter")
        window = self._cosmo.gal_cluster_window_Chi(Chis)
        I = step * matter_ps/(Chis)**2 * dChi * window ** 2
        return I.sum(axis=1)

    def get_phi_ps(self, ells, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100, extended=True, recalc_PK=False):
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
        if recalc_PK or self.weyl_PK is None:
            self.weyl_PK = self._get_PK("weyl", np.max(ells), Nchi)
        return self._Cl_phi(ells, Nchi, zmin, zmax, kmin, kmax, extended)

    def get_kappa_ps(self, ells, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100, extended=True, recalc_PK=False):
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
        if recalc_PK or self.weyl_PK is None:
            self.weyl_PK = self._get_PK("weyl", np.max(ells), Nchi)
        return self._Cl_kappa(ells, Nchi, zmin, zmax, kmin, kmax, extended)

    def get_kappa_ps_2source(self, ells, Chi_source1, Chi_source2=None, Nchi=100, kmin=0, kmax=100, extended=True, recalc_PK=False):
        """
        Returns the Limber approximated lensing convergence power spectrum for two source planes.

        Parameters
        ----------
        ells : int or float or ndarray
            Multipole moments at which to calculate the power spectrum.
        Chi_source1 : int or float or ndarray
            Comoving radial distance(s) of the first source plane(s) [Mpc]. Will be used for the first window function, and as the integral limit.
        Chi_source2 : None or int or float
            Comoving radial distance of the second source plane [Mpc]. Will be used for the second window function. If None, source plabe 1 will be used.
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
        if recalc_PK or self.weyl_PK is None:
            self.weyl_PK = self._get_PK("weyl", np.max(ells), Nchi)
        return self._Cl_kappa_2source(ells, Chi_source1, Chi_source2, Nchi, kmin, kmax, extended)

    def get_gal_kappa_ps(self, ells, Chi_source1=None, Nchi=100, kmin=0, kmax=100, extended=True, recalc_PK=False):
        """

        Parameters
        ----------
        ells
        Chi_source1
        Nchi
        kmin
        kmax
        extended
        recalc_matter_weyl

        Returns
        -------

        """
        if recalc_PK or self.matter_weyl_PK is None:
            self.matter_weyl_PK = self._get_PK("matter-weyl", np.max(ells), Nchi)
        return self._Cl_gal_kappa(ells, Chi_source1, Nchi, kmin, kmax, extended)

    def get_gal_ps(self, ells, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100, extended=True, recalc_PK=False):
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
        if recalc_PK or self.matter_PK is None:
            self.matter_PK = self._get_PK("matter", np.max(ells), Nchi)
        return self._Cl_gal(ells, Nchi, zmin, zmax, kmin, kmax, extended)


    def get_camb_postborn_omega_ps(self, ellmax=10000):
        """

        Parameters
        ----------
        ellmax

        Returns
        -------

        """
        return self._cosmo.get_postborn_omega_ps(2*ellmax)

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
        return 2 / (2 * ells + 1) * (Cl**2 + 0.5*(N0 * Cl))


if __name__ == "__main__":
    pass