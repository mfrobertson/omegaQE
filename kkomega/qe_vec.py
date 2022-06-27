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
            self.Cl_spline = None
            self.gradCl_spline = None
            self.N_spline = None


    def __init__(self):
        self._cosmo = Cosmology()
        self._noise = Noise()
        self.cmb = dict.fromkeys(self._cmb_types(), self.CMBsplines())
        self.initialise()
        self.C_inv_splines = self._build_cmb_Cov_inv_splines()

    def _cmb_types(self):
        types = np.char.array(list("TEB"))
        return (types[:, None] + types[None, :]).flatten()


    def _initialisation_check(self, typ, Lmax=4000):
        if not self.cmb[typ].initialised:
            self._initialise(typ, Lmax)

    def gmv_normalisation(self, Ls, curl):
        samp1 = np.arange(30, 40, 2)
        samp2 = np.logspace(1, 3, 200) * 4
        ells = np.concatenate((samp1, samp2))
        dTheta = 0.01
        thetas = np.arange(0, np.pi, dTheta)
        I1 = np.zeros(np.size(ells))
        typs = np.char.array(list("TEB"))
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
                w[Ls3 < 3] = 0
                w[Ls3 > 4000] = 0
                I2 = 0
                for i in typs:
                    for j in typs:
                        resp = self.response(i + j, L_vec, ell_vec, curl)
                        g = self.gmv_weight_function(i + j, L_vec, ell_vec, curl)
                        I2 += w * resp * g
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(30, 4000) / ((2 * np.pi) ** 2)
        return 1 / A

    def normalisation(self, typ, Ls, curl):
        samp1 = np.arange(30, 40, 2)
        samp2 = np.logspace(1, 3, 200) * 4
        ells = np.concatenate((samp1, samp2))
        dTheta = 0.01
        thetas = np.arange(0, np.pi, dTheta)
        I1 = np.zeros(np.size(ells))
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            for jjj, ell in enumerate(ells):
                ell_vec = vector.obj(rho=ell, phi=thetas)
                resp = self.response(typ, L_vec, ell_vec, curl)
                g = self.weight_function(typ, L_vec, ell_vec, curl)
                I2 = g * resp
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(30, 4000) / ((2 * np.pi) ** 2)
        return 1 / A

    def gmv_weight_function(self, typ, L_vec, ell_vec, curl):
        typs = np.char.array(list("TEB"))
        weight = 0
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        p = typ[0]
        q = typ[1]
        for i in typs:
            for j in typs:
                C_inv_ip_spline = self.get_cmb_Cov_inv_spline(i + p)
                C_inv_jq_spline = self.get_cmb_Cov_inv_spline(j + q)
                weight += self.response(i + j, L_vec, ell_vec, curl) * C_inv_ip_spline(ell) * C_inv_jq_spline(L3)
        return weight / 2

    def weight_function(self, typ, L_vec, ell_vec, curl):
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        denom = (self.cmb[typ1].Cl_spline(ell)+self.cmb[typ1].N_spline(ell))*(self.cmb[typ2].Cl_spline(L3)+self.cmb[typ2].N_spline(L3))
        fac = 0.5 if typ1 == typ2 else 1
        return fac*self.response(typ, L_vec, ell_vec, curl)/denom

    def response_phi(self, typ, L_vec, ell_vec):
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        w = np.ones(np.shape(L3))
        w[L3 < 3] = 0
        w[L3 > 4000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=ell_vec.deltaphi(L3_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=ell_vec.deltaphi(L3_vec))
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w*((L_vec @ ell_vec) * h1 * self.cmb["TE"].gradCl_spline(ell)) + ((L_vec @ L3_vec) * h2 * self.cmb["TE"].gradCl_spline(L3))
        elif typ == "EB":
            return w*((L_vec @ ell_vec) * h1 * self.cmb["EE"].gradCl_spline(ell)) + ((L_vec @ L3_vec) * h2 * self.cmb["BB"].gradCl_spline(L3))
        elif typ == "BE":
            return w*((L_vec @ ell_vec) * h1 * self.cmb["BB"].gradCl_spline(ell)) + ((L_vec @ L3_vec) * h2 * self.cmb["EE"].gradCl_spline(L3))
        return w*((L_vec @ ell_vec) * h1 * self.cmb[typ].gradCl_spline(ell)) + ((L_vec @ L3_vec) * h2 * self.cmb[typ].gradCl_spline(L3))

    def response(self, typ, L_vec, ell_vec, curl=True):
        if not curl:
            return self.response_phi(typ, L_vec, ell_vec)
        L = L_vec.rho
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        w = np.ones(np.shape(L3))
        w[L3 < 3] = 0
        w[L3 > 4000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=ell_vec.deltaphi(L3_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=ell_vec.deltaphi(L3_vec))
        self._initialisation_check(typ)
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w*L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self.cmb["TE"].gradCl_spline(ell) - h2 * self.cmb["TE"].gradCl_spline(L3))
        elif typ == "EB":
            return w*L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self.cmb["EE"].gradCl_spline(ell) - h2 * self.cmb["BB"].gradCl_spline(L3))
        elif typ == "BE":
            return w*L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self.cmb["BB"].gradCl_spline(ell) - h2 * self.cmb["EE"].gradCl_spline(L3))
        return w*L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self.cmb[typ].gradCl_spline(ell) - h2 * self.cmb[typ].gradCl_spline(L3))

    def geo_fac(self, typ, theta12):
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

    def _get_cmb_Cl_spline(self, typ):
        return self.cmb[typ].Cl_spline

    def _get_cmb_Cl(self, typ, Lmax=4000):
        Ls = np.arange(Lmax+1)
        return self.cmb[typ].Cl_spline(Ls)

    def _get_cmb_N(self, typ, Lmax=4000):
        Ls = np.arange(Lmax + 1)
        return self.cmb[typ].N_spline(Ls)

    def _get_cmb_cov(self, typ, Lmax=4000):
        Ls = np.arange(Lmax + 1)
        return self.cmb[typ].Cl_spline(Ls) + self.cmb[typ].N_spline(Ls)

    def get_cmb_cov_spline(self, typ, Lmax=4000):
        Ls = np.arange(Lmax + 1)
        return InterpolatedUnivariateSpline(Ls,self.cmb[typ].Cl_spline(Ls) + self.cmb[typ].N_spline(Ls))

    def _cmb_Cov_inv(self):
        typs = np.char.array(list("TEB"))
        C = typs[:, None] + typs[None, :]
        args = C.flatten()
        C_sym = Matrix(C)
        print(C_sym)
        C_inv = C_sym.inv()
        C_inv_func = lambdify(args, C_inv)
        Covs = [self._get_cmb_cov(arg) for arg in args]
        return C_inv_func(*Covs)

    def _build_cmb_Cov_inv_splines(self):
        C_inv = self._cmb_Cov_inv()
        C_inv_splines = np.empty((3,3), dtype=InterpolatedUnivariateSpline)
        for iii in range(3):
            for jjj in range(3):
                C_inv_ij = C_inv[iii,jjj]
                Ls = np.arange(np.size(C_inv_ij))
                C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[1:], C_inv_ij[1:])
        return C_inv_splines


    def get_cmb_Cov_inv_spline(self, typ):
        typs = np.char.array(list("TEB"))

        idx1 = np.where(typs == typ[0])[0][0]
        idx2 = np.where(typs == typ[1])[0][0]

        cov_inv = self.C_inv_splines[idx1][idx2]
        return cov_inv



    def _initialise(self, typ, Lmax=4000):
        if self.cmb[typ].initialised:
            return
        self.cmb[typ] = self.CMBsplines()
        Cl_lens = self._cosmo.get_lens_ps(typ, Lmax)
        Ls = np.arange(np.size(Cl_lens))
        Cl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], Cl_lens[2:])
        self.cmb[typ].Cl_spline = Cl_lens_spline
        gradCl_lens = self._cosmo.get_grad_lens_ps(typ, Lmax+int(Lmax/2))
        gradCl_lens_spline = InterpolatedUnivariateSpline(np.arange(np.size(gradCl_lens))[2:], gradCl_lens[2:])
        self.cmb[typ].gradCl_spline = gradCl_lens_spline
        N = self._noise.get_cmb_gaussian_N(typ, ellmax=Lmax)
        N_spline = InterpolatedUnivariateSpline(np.arange(np.size(N))[2:], N[2:])
        self.cmb[typ].N_spline = N_spline
        self.cmb[typ].initialised = True
        self._initialise(typ[::-1], Lmax)

    def initialise(self, Lmax=4000):
        typs = np.char.array(list("TEB"))
        C = np.triu(typs[:, None] + typs[None, :]).flatten()
        args = C[C!='']
        for arg in args:
            self._initialise(arg,Lmax)


if __name__ == '__main__': pass

