import numpy as np
from modecoupling import Modecoupling
from cache.tools import getFileSep
from scipy.interpolate import InterpolatedUnivariateSpline
import vector


class Postborn:


    def __init__(self):
        pass

    def _get_postborn_omega_ps(self, Ls, M_path, Nell_prim, Ntheta, M_ellmax, M_Nell, cmb):
        mode = Modecoupling()
        sep = getFileSep()
        mode_typ = "ww" if cmb else "rr"
        ells_sample = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "ells.npy")
        M = np.load(M_path + sep + mode_typ + sep + f"{M_ellmax}_{M_Nell}" + sep + "M.npy")
        M_spline = mode.spline(ells_sample, M)
        dTheta = np.pi / Ntheta
        thetas = np.linspace(dTheta, np.pi, Ntheta, dtype=float)
        samp1_1 = np.arange(3, 80, 1)
        samp2_1 = np.logspace(1, 3, Nell_prim - 77) * 8
        Lprims = np.concatenate((samp1_1, samp2_1))
        I = np.zeros(np.size(Ls))
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            I_tmp = np.zeros(np.size(Lprims))
            for jjj, Lprim in enumerate(Lprims):
                Lprim_vec = vector.obj(rho=Lprim, phi=thetas)
                Lprimprim_vec = L_vec - Lprim_vec
                Lprimprims = Lprimprim_vec.rho
                I_tmp[jjj] = np.sum(
                    2 * Lprim * dTheta * (L * Lprim * np.sin(thetas)) ** 2 * (Lprim_vec @ Lprimprim_vec) ** 2 / (
                                (Lprim + 0.5) ** 4 * (Lprimprims + 0.5) ** 4) * M_spline.ev(Lprim, Lprimprims))
            I[iii] = InterpolatedUnivariateSpline(Lprims, I_tmp).integral(3, 8000)
        return 4 * I / ((2 * np.pi) ** 2)


    def get_postborn_omega_ps(self, ells, M_path=None, Nell_prim=2000, Ntheta=1000, cmb=True):
        """

        Parameters
        ----------
        ells
        Nell_prim
        Ntheta

        Returns
        -------

        """
        return self._get_postborn_omega_ps(ells, M_path, Nell_prim, Ntheta, 8000, 100, cmb)