import camb
import numpy as np
import matplotlib.pyplot as plt

class Limber:
    """
    Calculates Limber approximated CMB lensing power spectra.

    Attributes
    ----------
    ellmax : int
        Maximum ell value over which to calculate potentials.
    Nchi : int
        Number of Chi values to use is integrals.
    ells_review : ndarray
        1D array of ell values over which Cl_phi_review is calculated for.
    Cl_phi_review : ndarray
        1D array of the lensing potential power spectrum calculated with reference to Weak Gravitational Lensing of the CMB (Lewis et al. 2006).
    """

    def __init__(self, ellmax=3000, Nchi=100):
        """
        Constructor.

        Parameters
        ----------
        ellmax : int
            Maximum ell value over which to calculate potentials.
        Nchi : int
            Number of Chi values to use is integrals.
        """
        self._pars = camb.CAMBparams()
        self._pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
        self._results = camb.get_background(self._pars)
        self._PK = self._get_weyl_PK()
        self.ellmax = ellmax
        self.Nchi = Nchi
        self.ells_review, self.Cl_phi_review = self._phi_ps_review(self.ellmax, self.Nchi)

    def _get_weyl_PK(self):
        PK_weyl = camb.get_matter_power_interpolator(self._pars, hubble_units=False, zmin=0, zmax=2000, kmax=100, k_hunit=False, var1=camb.model.Transfer_Weyl, var2=camb.model.Transfer_Weyl)
        return PK_weyl

    def get_weyl_ps(self, z, k, curly=False, scaled=True):
        """
        Returns the Weyl power spectrum.

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
        ps = self._PK.P(z, k, grid=False)
        if not scaled:
            ps *= k**-4
        if curly:
            return ps * k** 3 / (2 * np.pi ** 2)
        return ps

    def _get_chi_star(self):
        return self._get_eta0() - self._results.tau_maxvis

    def _get_eta0(self):
        return self._results.conformal_time(0)

    def _eta_to_z(self, eta):
        return self._results.redshift_at_conformal_time(eta)

    def _phi_ps_review(self, ellmax, Nchi):
        ells = np.arange(ellmax)
        Chi_str = self._get_chi_star()
        Chis = np.linspace(0, Chi_str, Nchi)
        dChi = Chis[1]
        eta0 = self._get_eta0()
        etas = eta0 - Chis
        zs = self._eta_to_z(etas)
        zs = zs[1:]
        Chis = Chis[1:]
        Cl_weyl = np.zeros(np.size(ells))
        for ell in ells[1:]:
            ks = ell / Chis
            win = (Chi_str - Chis) / (Chi_str * Chis)
            I = Chis * self.get_weyl_ps(zs, ks, curly=True, scaled=False) * dChi * win ** 2
            Cl_weyl[ell] = np.sum(I) / ell ** 3 * 8 * np.pi ** 2
        return ells[1:], Cl_weyl[1:]

if __name__ == "__main__":
    ks = np.logspace(-4, 2, 200)
    z = 20
    limber = Limber()
    ps = limber.get_weyl_ps(z, ks)

    plt.figure()
    plt.loglog(ks, ps)
    plt.show()

    ells = limber.ells_review
    Cl_weyl = limber.Cl_phi_review
    plt.figure()
    plt.loglog(ells, Cl_weyl*(ells*(ells + 1))**2/(2*np.pi))
    plt.show()