import numpy as np
from cosmology import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline
from noise import Noise
from sympy.matrices import Matrix
from sympy import lambdify
import vector


class QE:

    class CMBsplines:

        def __init__(self):
            self.initialised = False
            self.lenCl_spline = None
            self.unlenCl_spline = None
            self.gradCl_spline = None
            self.N_spline = None

    def __init__(self, deltaT=3, beam=3):
        self._cosmo = Cosmology()
        self._noise = Noise()
        self.cmb = dict.fromkeys(self._cmb_types(), self.CMBsplines())
        self.initialise(deltaT, beam)
        self._cov_inv_fields = "TEB"
        self._build_cmb_Cov_inv_splines(fields=self._cov_inv_fields)

    def _cmb_types(self):
        types = np.char.array(list("TEB"))
        return (types[:, None] + types[None, :]).flatten()

    def _get_cmb_cl(self, ells, fields, typ):
        if typ == "lensed":
            return self.cmb[fields].lenCl_spline(ells)
        elif typ == "unlensed":
            return self.cmb[fields].unlenCl_spline(ells)
        elif typ == "gradient":
            return self.cmb[fields].gradCl_spline(ells)

    def _initialisation_check(self, typ, Lmax=4000):
        if not self.cmb[typ].initialised:
            self._initialise(typ, Lmax)

    def gmv_normalisation(self, Ls, curl, fields="TEB", resp_ps="gradient"):
        """

        Parameters
        ----------
        Ls
        curl

        Returns
        -------

        """
        samp1 = np.arange(10, 40, 1)
        samp2 = np.logspace(1, 3, 300) * 4
        ells = np.concatenate((samp1, samp2))
        Ntheta = 1000
        dTheta = np.pi/Ntheta
        thetas = np.linspace(0, np.pi-dTheta, Ntheta)
        I1 = np.zeros(np.size(ells))
        typs = np.char.array(list(fields))
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            for jjj, ell in enumerate(ells):
                ell_vec = vector.obj(rho=ell, phi=thetas)
                Ls3_vec = L_vec - ell_vec
                Ls3 = Ls3_vec.rho
                w = np.ones(np.size(Ls3))
                w[Ls3 < 10] = 0
                w[Ls3 > 4000] = 0
                I2 = 0
                for i in typs:
                    for j in typs:
                        resp = self._response(i + j, L_vec, ell_vec, curl, resp_ps)
                        g = self.gmv_weight_function(i + j, L_vec, ell_vec, curl, fields, resp_ps)
                        I2 += w * resp * g
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(10, 4000) / ((2 * np.pi) ** 2)
        return 1 / A

    def normalisation(self, typ, Ls, curl, resp_ps="gradient"):
        """

        Parameters
        ----------
        typ
        Ls
        curl

        Returns
        -------

        """
        if typ == "gmv":
            return self.gmv_normalisation(Ls, curl, resp_ps=resp_ps)
        samp1 = np.arange(10, 40, 1)
        samp2 = np.logspace(1, 3, 300) * 4
        ells = np.concatenate((samp1, samp2))
        Ntheta = 1000
        dTheta = np.pi / Ntheta
        thetas = np.linspace(0, np.pi-dTheta, Ntheta)
        I1 = np.zeros(np.size(ells))
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            for jjj, ell in enumerate(ells):
                ell_vec = vector.obj(rho=ell, phi=thetas)
                resp = self._response(typ, L_vec, ell_vec, curl, resp_ps)
                g = self.weight_function(typ, L_vec, ell_vec, curl, gmv=False, resp_ps=resp_ps)
                I2 = g * resp
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(10, 4000) / ((2 * np.pi) ** 2)
        return 1 / A

    def gmv_weight_function(self, typ, L_vec, ell_vec, curl, fields="TEB", resp_ps="gradient"):
        """

        Parameters
        ----------
        typ
        L_vec
        ell_vec
        curl

        Returns
        -------

        """
        weight = 0
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        p = typ[0]
        q = typ[1]
        typs = np.char.array(list(fields))
        XYs = np.triu(typs[:, None] + typs[None, :]).flatten()
        XYs = XYs[XYs != ""]
        XYs = XYs[XYs != "BB"]
        for ij in XYs:
            i = ij[0]
            j = ij[1]
            fac = 1 if i == j else 2
            C_inv_ip_spline = self._get_cmb_Cov_inv_spline(i + p, fields)
            C_inv_jq_spline = self._get_cmb_Cov_inv_spline(j + q, fields)
            weight += fac * self._response(i + j, L_vec, ell_vec, curl, resp_ps) * C_inv_ip_spline(ell) * C_inv_jq_spline(L3)
        return weight / 2


    def weight_function(self, typ, L_vec, ell_vec, curl, gmv=False, fields="TEB", resp_ps="gradient"):
        """

        Parameters
        ----------
        typ
        L_vec
        ell_vec
        curl
        gmv

        Returns
        -------

        """
        if gmv:
            return self.gmv_weight_function(typ, L_vec, ell_vec, curl, fields, resp_ps)
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        denom = (self.cmb[typ1].lenCl_spline(ell)+self.cmb[typ1].N_spline(ell))*(self.cmb[typ2].lenCl_spline(L3)+self.cmb[typ2].N_spline(L3))
        fac = 0.5 if typ1 == typ2 else 1
        return fac*self._response(typ, L_vec, ell_vec, curl, resp_ps)/denom

    def _response_phi(self, typ, L_vec, ell_vec, cl="gradient"):
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        w = np.ones(np.shape(L3))
        w[L3 < 10] = 0
        w[L3 > 4000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=ell_vec.deltaphi(L3_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=ell_vec.deltaphi(L3_vec))
        # self._initialisation_check(typ)
        if typ == "BT" or typ == "TB" or typ == "TE" or typ == "ET":
            typ1 = typ2 = "TE"
        else:
            typ1 = typ[0] + typ[0]
            typ2 = typ[1] + typ[1]
        return w*((L_vec @ ell_vec) * h1 * self._get_cmb_cl(ell, typ1, cl)) + ((L_vec @ L3_vec) * h2 * self._get_cmb_cl(L3, typ2, cl))

    def _response(self, typ, L_vec, ell_vec, curl=True, cl="gradient"):
        if not curl:
            return self._response_phi(typ, L_vec, ell_vec, cl)
        L = L_vec.rho
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        w = np.ones(np.shape(L3))
        w[L3 < 10] = 0
        w[L3 > 4000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=ell_vec.deltaphi(L3_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=ell_vec.deltaphi(L3_vec))
        # self._initialisation_check(typ)
        # self._initialisation_check(typ)
        if typ == "BT" or typ == "TB" or typ == "TE" or typ == "ET":
            typ1 = typ2 = "TE"
        else:
            typ1 = typ[0] + typ[0]
            typ2 = typ[1] + typ[1]
        return w*L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self._get_cmb_cl(ell, typ1, cl) - h2 * self._get_cmb_cl(L3, typ2, cl))

    def geo_fac(self, typ, theta12):
        """

        Parameters
        ----------
        typ
        theta12

        Returns
        -------

        """
        shape = np.shape(theta12)
        if typ == "T":
            return self._geo_fac_T(shape)
        elif typ == "E":
            return self._geo_fac_E(theta12)
        elif typ == "B":
            return self._geo_fac_B(theta12)

    def _geo_fac_T(self, shape):
        return np.ones(shape)

    def _geo_fac_E(self, theta12):
        return np.cos(2*(theta12))

    def _geo_fac_B(self, theta12):
        return np.sin(2*(theta12))

    def _get_response_geo_fac(self, typ, num, theta12):
        shape = np.shape(theta12)
        if typ == "TT":
            return self._geo_fac_T(shape)
        elif typ == "EE" or typ == "BB":
            return self._geo_fac_E(theta12)
        elif typ == "EB" or typ == "BE":
            return self._geo_fac_B(theta12)
        elif typ == "TE":
            if num == 1:
                return self._geo_fac_E(theta12)
            return self._geo_fac_T(shape)
        elif typ == "ET":
            if num == 1:
                return self._geo_fac_T(shape)
            return self._geo_fac_E(theta12)
        elif typ == "TB":
            if num == 1:
                return self._geo_fac_B(theta12)
            return 0
        elif typ == "BT":
            if num == 1:
                return 0
            return self._geo_fac_B(theta12)
        else:
            raise ValueError(f"Type {typ} does not exist.")

    def _get_cmb_cov(self, typ, Lmax=4000):
        Ls = np.arange(Lmax + 1)
        return self.cmb[typ].lenCl_spline(Ls) + self.cmb[typ].N_spline(Ls)

    def _cmb_Cov_inv(self, fields):
        typs = np.char.array(list(fields))
        C = typs[:, None] + typs[None, :]
        args = C.flatten()
        C_sym = Matrix(C)
        C_inv = C_sym.inv()
        C_inv_func = lambdify(args, C_inv)
        Covs = [self._get_cmb_cov(arg) for arg in args]
        return C_inv_func(*Covs)

    def _build_cmb_Cov_inv_splines(self, fields):
        self._cov_inv_fields = fields
        N_fields = len(self._cov_inv_fields)
        C_inv = self._cmb_Cov_inv(fields)
        C_inv_splines = np.empty((N_fields,N_fields), dtype=InterpolatedUnivariateSpline)
        for iii in range(N_fields):
            for jjj in range(N_fields):
                C_inv_ij = C_inv[iii,jjj]
                Ls = np.arange(np.size(C_inv_ij))
                C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[2:], C_inv_ij[2:])
        self.C_inv_splines = C_inv_splines


    def _get_cmb_Cov_inv_spline(self, typ, fields="TEB"):
        if fields != self._cov_inv_fields:
            self._build_cmb_Cov_inv_splines(fields)
        typs = np.char.array(list(fields))

        idx1 = np.where(typs == typ[0])[0][0]
        idx2 = np.where(typs == typ[1])[0][0]

        cov_inv = self.C_inv_splines[idx1][idx2]
        return cov_inv



    def _initialise(self, typ, deltaT, beam, Lmax=4000):
        if self.cmb[typ].initialised:
            return
        self.cmb[typ] = self.CMBsplines()
        Cl_lens = self._cosmo.get_lens_ps(typ, 2*Lmax)
        Ls = np.arange(np.size(Cl_lens))
        Cl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], Cl_lens[2:])
        self.cmb[typ].lenCl_spline = Cl_lens_spline

        Cl_unlens = self._cosmo.get_unlens_ps(typ, 2*Lmax)
        Ls = np.arange(np.size(Cl_unlens))
        Cl_unlens_spline = InterpolatedUnivariateSpline(Ls[2:], Cl_unlens[2:])
        self.cmb[typ].unlenCl_spline = Cl_unlens_spline

        gradCl_lens = self._cosmo.get_grad_lens_ps(typ, 2*Lmax)
        Ls = np.arange(np.size(gradCl_lens))
        gradCl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], gradCl_lens[2:])
        self.cmb[typ].gradCl_spline = gradCl_lens_spline

        N = self._noise.get_cmb_gaussian_N(typ, ellmax=Lmax, deltaT=deltaT, beam=beam)
        N_spline = InterpolatedUnivariateSpline(np.arange(np.size(N))[2:], N[2:])
        self.cmb[typ].N_spline = N_spline
        self.cmb[typ].initialised = True
        self._initialise(typ[::-1], deltaT, beam, Lmax)

    def initialise(self, deltaT, beam, Lmax=4000):
        """

        Parameters
        ----------
        Lmax

        Returns
        -------

        """
        typs = np.char.array(list("TEB"))
        C = np.triu(typs[:, None] + typs[None, :]).flatten()
        args = C[C!='']
        for arg in args:
            self._initialise(arg, deltaT, beam, Lmax)


if __name__ == '__main__': pass

