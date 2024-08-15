from omegaqe.powerspectra import Powerspectra
from omegaqe.tools import maths
import numpy as np


class Modecoupling:
    """
    Calculates the mode-coupling matrix.

    Attributes
    ----------
    weyl_PK : object
        CAMB RectBivariateSpline PK object as the interpolator for the Weyl potential power spectrum.
    """

    def __init__(self, powerspectra=None, matter_PK=None):
        """
        Constructor

        """
        self._powerspectra = Powerspectra() if powerspectra is None else powerspectra
        self._cosmo = self._powerspectra.cosmo
        self.matter_PK = self._cosmo.get_matter_PK(typ="matter") if matter_PK is None else matter_PK
        self.binned_gal_types = list("abcdef")
        self.use_LSST_abcde = False

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

    def _get_window(self, typ, chi, nu, gal_bins, gal_distro="LSST_gold", chi_max=None):
        cmb_lens_window = self._cosmo.cmb_lens_window_matter(chi, self._cosmo.get_chi_star())
        if typ == "k":
            return cmb_lens_window
        elif typ == "g":
            return self._cosmo.gal_window_Chi(chi, typ=gal_distro)
        elif typ == "I":
           return self._cosmo.cib_window_Chi(chi, nu=nu)
        elif typ in ("r", "s"):
            return self._cosmo.gal_lens_window_matter(chi, chi_max)
        elif typ in self.binned_gal_types:
            index = 2*(ord(typ[0]) - ord("a"))
            if self.use_LSST_abcde:
                gal_distro = f"LSST_{typ[0]}"
                gal_window = self._cosmo.gal_window_Chi(chi, typ=gal_distro)
            else:
                gal_window = self._cosmo.gal_window_Chi(chi, zmin=gal_bins[index], zmax=gal_bins[index+1], typ=gal_distro)
            return gal_window
        raise ValueError(f"Unknown window type: {typ}.")

    def _integral_prep(self, Nchi, zmin, zmax, typ, nu, gal_bins, gal_distro="LSST_gold", sec_order_var="k"):
        Chi_min = self._cosmo.z_to_Chi(zmin)
        if zmax is not None:
            Chi_max = self._cosmo.z_to_Chi(zmax)
        else:
            Chi_max = self._cosmo.get_chi_star()
        Chis = np.linspace(Chi_min, Chi_max, Nchi)[1:]
        dChi = Chis[1] - Chis[0]
        zs = self._cosmo.Chi_to_z(Chis)
        if sec_order_var in ("k", "w", "r"):
            win1 = self._get_window("k", Chis, nu, gal_bins, gal_distro)
        else:
            win1 = self._get_window(sec_order_var, Chis, nu, gal_bins, gal_distro, Chi_max)
        win2 = self._get_window(typ[0], Chis, nu, gal_bins, gal_distro)
        return zs, Chis, dChi, win1, win2

    def _get_ps(self, ells, Chis, Chi_source2, typ, nu, gal_bins, recalc_PK, gal_distro="LSST_gold"):
        if typ[1] == "k":
            return self._powerspectra.get_kappa_ps_2source(ells, Chis, Chi_source2, recalc_PK=recalc_PK, use_weyl=False)
        if typ[1] == "r":
            return self._powerspectra.get_gal_lens_ps_2source(ells, Chis, Chi_source2, recalc_PK=recalc_PK, use_weyl=False)
        if typ[1] == "s":
            return self._powerspectra.get_gal_lens_kappa_ps(ells, Chis, recalc_PK=recalc_PK, use_weyl=False)
        if typ[1] == "g":
            return self._powerspectra.get_gal_kappa_ps(ells, Chis, recalc_PK=recalc_PK, gal_distro=gal_distro, use_weyl=False)
        if typ[1] in self.binned_gal_types:
            index = 2*(ord(typ[1]) - ord("a"))
            if self.use_LSST_abcde:
                gal_distro = f"LSST_{typ[1]}"
                return self._powerspectra.get_gal_kappa_ps(ells, Chis, recalc_PK=recalc_PK, gal_distro=gal_distro, use_weyl=False)
            return self._powerspectra.get_gal_kappa_ps(ells, Chis, recalc_PK=recalc_PK, gal_win_zmin=gal_bins[index], gal_win_zmax=gal_bins[index+1], gal_distro=gal_distro, use_weyl=False)
        if typ[1] == "I":
            return self._powerspectra.get_cib_kappa_ps(ells, nu=nu, Chi_source1=Chis, recalc_PK=recalc_PK, use_weyl=False)

    def _get_matter_ps(self, zs, ks):
        return self._cosmo.get_matter_ps(self.matter_PK, zs, ks, curly=False, weyl_scaled=False, typ="matter")

    def _components(self, ells1, ells2, typ, star, Nchi, kmin, kmax, zmin, zmax, nu, gal_bins, extended, recalc_PK, gal_distro="LSST_gold", sec_order_var="k"):
        zs, Chis, dChi, win1, win2 = self._integral_prep(Nchi, zmin, zmax, typ, nu, gal_bins, gal_distro=gal_distro, sec_order_var=sec_order_var)
        Nells1 = np.size(ells1)
        ells1_vec = self._vectorise_ells(ells1, zs.ndim)
        zs = self._vectorise_zs(zs, Nells1)
        if star:
            Chi_source2 = self._cosmo.get_chi_star()
        else:
            Chi_source2 = None
        Cl = self._get_ps(ells2, Chis, Chi_source2, typ, nu, gal_bins, recalc_PK=recalc_PK, gal_distro=gal_distro)
        if recalc_PK:
            self.weyl_PK = self._powerspectra.weyl_PK
            self.matter_weyl_PK = self._powerspectra.matter_weyl_PK
            self.matter_PK = self._powerspectra.matter_PK
        if extended:
            ks = (ells1_vec + 0.5) / Chis
        else:
            ks = ells1_vec / Chis
        step = maths.rectangular_pulse_steps(ks, kmin, kmax)
        matter_ps = self._get_matter_ps(zs, ks)
        I = step * matter_ps / Chis ** 2 * dChi * win1 * win2 * Cl
        return I.sum(axis=1)

    def _matrix(self, ells1, ells2, typ, star, Nchi, kmin, kmax, zmin, zmax, nu, gal_bins, extended, recalc_weyl, gal_distro="LSST_gold", sec_order_var="k"):
        M = np.ones((np.size(ells1), np.size(ells2)))
        for iii, ell1 in enumerate(ells1):
            M[iii, :] = self._components(np.ones(np.size(ells2))*ell1, ells2, typ, star, Nchi, kmin, kmax, zmin, zmax, nu, gal_bins, extended, recalc_weyl, gal_distro=gal_distro, sec_order_var=sec_order_var)
        return M

    def _check_type(self, typ):
        typs = self.get_M_types()
        if typ not in typs:
            raise ValueError(f"Modecoupling type {typ} not from accepted types: {typs}")

    def check_type(self, typ):
        try:
            self._check_type(typ)
        except:
            return False
        return True

    def get_M_types(self):
        """

        Returns
        -------

        """
        observables = np.char.array(list("kgIwrs")+self.binned_gal_types)
        return (observables[:, None] + observables[None, :]).flatten()


    def components(self, ells1, ells2, typ="kk", star=True, Nchi=100, kmin=0, kmax=100, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), extended=False, recalc_PK=False, gal_distro="LSST_gold", sec_order_var="k"):
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
        recalc_PK : bool
            Recalculate the matter power spectrum interpolator optimised for this particular calculation given the supplied inputs.

        Returns
        -------
        ndarray
            1D array of the matrix components at [ells, ells2].
        """
        self._check_type(typ)
        if typ == "ww":
            typ = "kk"
            star = False
        elif typ == "rr":
            star = False
        return self._components(ells1, ells2, typ, star, Nchi, kmin, kmax, zmin, zmax, nu, gal_bins, extended, recalc_PK, gal_distro=gal_distro, sec_order_var=sec_order_var)

    def spline(self, ells_sample=None, M_matrix=None, typ = "kk", star=True, Nchi=100, kmin=0, kmax=100, zmin=0, zmax=None, nu=353e9, gal_bins=(None,None,None,None), extended=False, recalc_PK=False, gal_distro="LSST_gold", sec_order_var="k"):
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
        recalc_PK : bool
            Recalculate the matter power spectrum interpolator optimised for this particular calculation given the supplied inputs.

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
        self._check_type(typ)
        M = self._matrix(ells_sample, ells_sample, typ, star, Nchi, kmin, kmax, zmin, zmax, nu, gal_bins, extended, recalc_PK, gal_distro=gal_distro, sec_order_var=sec_order_var)
        return RectBivariateSpline(ells_sample, ells_sample, M)

    def generate_sample_ells(self, ellmax=5000, Nells=100):
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
