import camb
import numpy as np
import matplotlib.pyplot as plt
from cosmology import Cosmology

class Limber:
    """
    Calculates Limber approximated CMB lensing power spectra.

    Attributes
    ----------
    ellmax : int
        Maximum ell value over which to calculate potentials.
    Nchi : int
        Number of Chi values to use during integration.
    ells : ndarray
        1D array of ell values over which Cl_phi is calculated for.
    Cl_phi : ndarray
        1D array of the lensing potential power spectrum calculated with reference to Weak Gravitational Lensing of the CMB (Lewis et al. 2006).
    ells_extended : ndarray
        1D array of ell values over which Cl_phi_extended is calculated for.
    Cl_phi_extended : ndarray
        1D array of the lensing potential power spectrum calculated with extended Limber approximation ( i.e. k=(ell+0.5)/Chi).
    """

    def __init__(self, ellmax=3000, Nchi=100, compute=True):
        """
        Constructor.

        Parameters
        ----------
        ellmax : int
            Maximum ell value over which to calculate potentials.
        Nchi : int
            Number of Chi values to use in integrals.
        compute : bool
            Calculate interpolated Weyl potential and compute the limber approximated lensing potential power spectra.
        """
        self.ellmax = ellmax
        self.Nchi = Nchi
        self._cosmo = Cosmology()

        if compute:
            self.compute()

    def compute(self):
        self._PK = self._get_weyl_PK()
        self.ells, self.Cl_phi = self._phi_ps(self.ellmax, self.Nchi)
        self.ells_extended, self.Cl_phi_extended = self._phi_ps(self.ellmax, self.Nchi, extended=True)

    def _get_weyl_PK(self):
        zbuffer = 100
        zmax = self._cosmo.eta_to_z(self._cosmo.get_eta0() - self._cosmo.get_chi_star()) + zbuffer
        kbuffer = 10
        kmax = self.ellmax * self.Nchi/self._cosmo.get_chi_star() + kbuffer
        return self._cosmo.get_weyl_PK(kmax, zmax)

    def _phi_ps(self, ellmax, Nchi, zmin=0, zmax=None, kmin=0, kmax=100, extended=False):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        Chi_str = self._cosmo.get_chi_star()
        if zmax is None:
            Chi_max = Chi_str
        else:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        Chis = np.linspace(Chi_min, Chi_max, Nchi)
        dChi = Chis[1] - Chis[0]
        eta0 = self._cosmo.get_eta0()
        etas = eta0 - Chis
        zs = self._cosmo.eta_to_z(etas)
        zs = zs[1:]
        Chis = Chis[1:]
        step = np.ones(Chis.shape)
        win = self._cosmo.window(Chis, Chi_max)
        ells = np.arange(ellmax)
        Cl_weyl = np.zeros(np.size(ells))
        for ell in ells[1:]:
            if extended:
                ks = (ell + 0.5)/ Chis
            else:
                ks = ell / Chis
            step[:] = 1
            step[ks < kmin] = 0
            step[ks > kmax] = 0
            weyl_ps = self._cosmo.get_weyl_ps(self._PK, zs, ks, curly=True, scaled=False)
            I = step * Chis * weyl_ps * dChi * win ** 2
            if extended:
                Cl_weyl[ell] = np.sum(I) / (ell+0.5)** 3 * 8 * np.pi ** 2
            else:
                Cl_weyl[ell] = np.sum(I) / ell ** 3 * 8 * np.pi ** 2
        return ells[1:], Cl_weyl[1:]

    def get_phi_ps(self, zmin=0, zmax=None, kmin=0, kmax=100, extended=True):
        """
        Return the Limber approximated lensing potential power spectrum.

        Parameters
        ----------
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

        Returns
        -------
        2-tuple
            The first object in the tuple in a 1D ndarry of the ell values. The second object is a 1D ndarray of the calculated lensing potential.
        """
        if (zmin, zmax, kmin, kmax) == (0, None, 0, 100):
            if extended:
                return self.ells_extended, self.Cl_phi_extended
            else:
                return self.ells, self.Cl_phi
        return self._phi_ps(self.ellmax, self.Nchi, zmin, zmax, kmin, kmax, extended)

if __name__ == "__main__":
    ks = np.logspace(-4, 2, 200)
    z = 20
    limber = Limber()

    ells = limber.ells
    Cl_weyl = limber.Cl_phi
    plt.figure()
    plt.loglog(ells, Cl_weyl*(ells*(ells + 1))**2/(2*np.pi))
    plt.show()