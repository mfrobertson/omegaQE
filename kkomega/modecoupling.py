from limber import Limber
from cosmology import Cosmology
import numpy as np

class modecoupling:

    def __init__(self):
        self._cosmo = Cosmology()
        self._weyl_PK = self._get_weyl_PK()

    def _get_weyl_PK(self):
        zbuffer = 100
        zmax = self._cosmo.eta_to_z(self._cosmo.get_eta0() - self._cosmo.get_chi_star()) + zbuffer
        kbuffer = 10
        kmax = self.ellmax * self.Nchi/self._cosmo.get_chi_star() + kbuffer
        return self._cosmo.get_weyl_PK(kmax, zmax)

    def compute(self, ell1, ell2, Nchi=100, zmin=0, zmax=None, kmin=0, kmax=100):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        Chi_str = self._cosmo.get_chi_star()
        if zmax is None:
            Chi_max = Chi_str
        else:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        Chis = np.linspace(Chi_min, Chi_max, Nchi)
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        zs = zs[1:]
        Chis = Chis[1:]
        step = np.ones(Chis.shape)
        win = self._cosmo.window(Chis, Chi_max)
        ks1 = ell1 / Chis
        step[:] = 1
        step[ks1 < kmin] = 0
        step[ks1 > kmax] = 0
        weyl_ps = self._cosmo.get_weyl_ps(self._weyl_PK, zs, ks1, curly=True, scaled=False)
        I = step * weyl_ps * dChi * (win/Chis)**2
        return ell1**4 * I
