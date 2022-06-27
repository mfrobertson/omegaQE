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

    def _get_third_L(self, L1, L2, theta):
        return np.sqrt(L1**2 + L2**2 - (2*L1*L2*np.cos(theta).astype("double"))).astype("double")

    def _initialisation_check(self, typ, Lmax=4000):
        if not self.cmb[typ].initialised:
            self._initialise(typ, Lmax)


    def response_phi(self, typ, l, L, theta):
        L4 = self._get_third_L(l,L,theta)
        w = np.ones(np.shape(L4))
        w[L4 < 3] = 0
        w[L4 > 6000] = 0
        theta4L = np.arcsin(l*np.sin(theta)/L4)
        thetal4 = 2*np.pi - theta4L - theta
        h1 = self._get_response_geo_fac(typ, 1, theta12=thetal4)
        h2 = self._get_response_geo_fac(typ, 2, theta12=thetal4)
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w * (L * l * np.cos(theta) * h1 * self.cmb["TE"].gradCl_spline(l)) + (L * L4 * np.cos(theta4L) * h2 * self.cmb["TE"].gradCl_spline(L4))
        elif typ == "EB":
            return w * (L * l * np.cos(theta) * h1 * self.cmb["EE"].gradCl_spline(l)) + (L * L4 * np.cos(theta4L) * h2 * self.cmb["BB"].gradCl_spline(L4))
        elif typ == "BE":
            return w * (L * l * np.cos(theta) * h1 * self.cmb["BB"].gradCl_spline(l)) + (L * L4 * np.cos(theta4L) * h2 * self.cmb["EE"].gradCl_spline(L4))
        return w*(L * l * np.cos(theta)*h1*self.cmb[typ].gradCl_spline(l)) + (L * L4 * np.cos(theta4L) * h2*self.cmb[typ].gradCl_spline(L4))

    def response(self, typ, l, L, theta, curl):
        if not curl:
            return self.response_phi(typ, l, L, theta)
        L4 = self._get_third_L(l, L, theta)
        w = np.ones(np.shape(L4))
        w[L4 < 3] = 0
        w[L4 > 6000] = 0
        theta4L = np.arcsin(l * np.sin(theta) / L4)
        thetal4 = 2 * np.pi - theta4L - theta
        h1 = self._get_response_geo_fac(typ, 1, theta12=thetal4)
        h2 = self._get_response_geo_fac(typ, 2, theta12=thetal4)
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w * L * l * np.sin(theta) * (h1 * self.cmb["TE"].gradCl_spline(l) - h2 * self.cmb["TE"].gradCl_spline(L4))
        elif typ == "EB":
            return w * L * l * np.sin(theta) * (h1 * self.cmb["EE"].gradCl_spline(l) - h2 * self.cmb["BB"].gradCl_spline(L4))
        elif typ == "BE":
            return w * L * l * np.sin(theta) * (h1 * self.cmb["BB"].gradCl_spline(l) - h2 * self.cmb["EE"].gradCl_spline(L4))
        # print(f"theta : {theta}")
        # print(f"L4: {L4}")
        return w * L * l * np.sin(theta) * (h1 * self.cmb[typ].gradCl_spline(l) - h2 * self.cmb[typ].gradCl_spline(L4))

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

    def _get_cmb_cov_spline(self, typ, Lmax=4000):
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

    def _gmv_weight_denom(self, typ, L1, L2, theta, C_TT, C_EE, C_TE):
        L3 = self._get_third_L(L1, L2, theta)
        D1 = C_TT(L1)*C_EE(L1) - C_TE(L1)**2
        D2 = C_TT(L3)*C_EE(L3) - C_TE(L3)**2
        if typ[1] == "B":
            return 2 * D1 * self._get_cmb_cov_spline("BB")(L3)
        elif typ[0] == "B":
            return 2 * D2 * self._get_cmb_cov_spline("BB")(L1)
        return 2*D1*D2

    def _gmv_weight_numer(self, typ, L1, L2, theta, curl, C_TT, C_EE, C_TE):
        L3 = self._get_third_L(L1, L2, theta)
        f_TT = self.response("TT", L1, L2, theta, curl)
        f_TE = self.response("TE", L1, L2, theta, curl)
        f_ET = self.response("ET", L1, L2, theta, curl)
        f_EE = self.response("EE", L1, L2, theta, curl)
        f_TB = self.response("TB", L1, L2, theta, curl)
        f_EB = self.response("EB", L1, L2, theta, curl)
        f_BT = self.response("BT", L1, L2, theta, curl)
        f_BE = self.response("BE", L1, L2, theta, curl)
        if typ == "TT":
            return C_EE(L1)*C_EE(L3)*f_TT + C_TE(L1)*C_TE(L3)*f_EE - C_EE(L1)*C_TE(L3)*f_TE - C_TE(L1)*C_EE(L3)*f_ET
        elif typ == "EE":
            return C_TE(L1)*C_TE(L3)*f_TT + C_TT(L1)*C_TT(L3)*f_EE - C_TE(L1)*C_TT(L3)*f_TE - C_TT(L1)*C_TE(L3)*f_ET
        elif typ == "TE":
            # return -C_TE(L1)*C_EE(L3)*f_TT - C_TE(L1)*C_TT(L3)*f_EE + C_EE(L1)*C_TT(L3)*f_TE + C_TE(L1)*C_TE(L3)*f_ET
            return -C_EE(L1) * C_TE(L3) * f_TT - C_TE(L1) * C_TT(L3) * f_EE + C_EE(L1) * C_TT(L3) * f_TE + C_TE(L1) * C_TE(L3) * f_ET
        elif typ == "ET":
            # return -C_TE(L3)*C_EE(L1)*f_TT - C_TE(L3)*C_TT(L1)*f_EE + C_EE(L3)*C_TT(L1)*f_ET + C_TE(L3)*C_TE(L1)*f_TE
            return -C_EE(L3) * C_TE(L1) * f_TT - C_TE(L3) * C_TT(L1) * f_EE + C_EE(L3) * C_TT(L1) * f_ET + C_TE(L3) * C_TE(L1) * f_TE
        elif typ == "TB":
            return C_EE(L1)*f_TB - C_TE(L1)*f_EB
        elif typ == "BT":
            return C_EE(L3)*f_BT - C_TE(L3)*f_BE
        elif typ == "EB":
            return -C_TE(L1)*f_TB + C_TT(L1)*f_EB
        elif typ == "BE":
            return -C_TE(L3)*f_BT + C_TT(L3)*f_BE
        elif typ == "BB":
            return np.zeros(np.shape(L1))
        else:
            raise ValueError(f"Type {typ} does not exist.")


    def _get_cmb_Cov_inv_spline(self, typ):
        typs = np.char.array(list("TEB"))

        idx1 = np.where(typs == typ[0])[0][0]
        idx2 = np.where(typs == typ[1])[0][0]

        cov_inv = self.C_inv_splines[idx1][idx2]
        return cov_inv

    def gmv_weight_function(self, typ, L1, L2, theta, curl, explicit=True):
        if explicit:
            C_TT = self._get_cmb_cov_spline("TT")
            C_EE = self._get_cmb_cov_spline("EE")
            C_TE = self._get_cmb_cov_spline("TE")
            denom = self._gmv_weight_denom(typ, L1, L2, theta, C_TT, C_EE, C_TE)
            numer = self._gmv_weight_numer(typ, L1, L2, theta, curl, C_TT, C_EE, C_TE)
            return numer/denom
        typs = np.char.array(list("TEB"))
        C = typs[:, None] + typs[None, :]
        args = C.flatten()
        weight = 0
        L3 = self._get_third_L(L1, L2, theta)
        for arg in args:
            ip = arg[0]+typ[0]
            jq = arg[1]+typ[1]
            C_inv_ip_spline = self._get_cmb_Cov_inv_spline(ip)
            C_inv_jq_spline = self._get_cmb_Cov_inv_spline(jq)
            weight += self.response(arg, L1, L2, theta, curl)*C_inv_ip_spline(L1)*C_inv_jq_spline(L3)
        return weight/2

    def weight_function_vector(self, typ, L3_vec, L_vec, curl):
        L = L_vec.rho
        L4_vec = L_vec - L3_vec
        L4 = L4_vec.rho
        w = np.ones(np.shape(L4))
        w[L4<3] = 0
        w[L4 > 6000] = 0
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        denom = 2*(self.cmb[typ1].Cl_spline(L)+self.cmb[typ1].N_spline(L))*(self.cmb[typ2].Cl_spline(L4)+self.cmb[typ2].N_spline(L4))
        return w*self.response_vector(typ, L_vec, L3_vec, curl)/denom

    def response_vector_phi(self, typ, L3_vec, L_vec):
        L = L_vec.rho
        L3 = L3_vec.rho
        L4_vec = L_vec - L3_vec
        L4 = L4_vec.rho
        w = np.ones(np.shape(L4))
        w[L4 < 3] = 0
        w[L4 > 6000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=L3_vec.deltaphi(L4_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=L3_vec.deltaphi(L4_vec))
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w * (L * L3 * np.cos(L_vec.deltaphi(L3_vec)) * h1 * self.cmb["TE"].gradCl_spline(L3)) + (L * L4 * np.cos(L_vec.deltaphi(L4_vec)) * h2 * self.cmb["TE"].gradCl_spline(L4))
        elif typ == "EB":
            return w * (L * L3 * np.cos(L_vec.deltaphi(L3_vec)) * h1 * self.cmb["EE"].gradCl_spline(L3)) + (L * L4 * np.cos(L_vec.deltaphi(L4_vec)) * h2 * self.cmb["BB"].gradCl_spline(L4))
        elif typ == "BE":
            return w * (L * L3 * np.cos(L_vec.deltaphi(L3_vec)) * h1 * self.cmb["BB"].gradCl_spline(L3)) + (L * L4 * np.cos(L_vec.deltaphi(L4_vec)) * h2 * self.cmb["EE"].gradCl_spline(L4))
        return w*(L * L3 * np.cos(L_vec.deltaphi(L3_vec))*h1*self.cmb[typ].gradCl_spline(L3)) + (L * L4 * np.cos(L_vec.deltaphi(L4_vec)) * h2*self.cmb[typ].gradCl_spline(L4))

    def response_vector(self, typ, L3_vec, L_vec, curl):
        if not curl:
            return self.response_vector_phi(typ, L3_vec, L_vec)
        L = L_vec.rho
        L3 = L3_vec.rho
        L4_vec = L_vec - L3_vec
        L4 = L4_vec.rho
        w = np.ones(np.shape(L4))
        w[L4 < 3] = 0
        w[L4 > 6000] = 0
        h1 = self._get_response_geo_fac(typ, 1, theta12=L3_vec.deltaphi(L4_vec))
        h2 = self._get_response_geo_fac(typ, 2, theta12=L3_vec.deltaphi(L4_vec))
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w * L * L3 * np.sin(L_vec.deltaphi(L3_vec)) * (h1 * self.cmb["TE"].gradCl_spline(L3) - h2 * self.cmb["TE"].gradCl_spline(L4))
        elif typ == "EB":
            return w * L * L3 * np.sin(L_vec.deltaphi(L3_vec)) * (h1 * self.cmb["EE"].gradCl_spline(L3) - h2 * self.cmb["BB"].gradCl_spline(L4))
        elif typ == "BE":
            return w * L * L3 * np.sin(L_vec.deltaphi(L3_vec)) * (h1 * self.cmb["BB"].gradCl_spline(L3) - h2 * self.cmb["EE"].gradCl_spline(L4))
        # print(f"theta: {L_vec.deltaphi(L3_vec)}")
        # print(f"L4 (vector): {L4}")
        return w * L * L3 * np.sin(L_vec.deltaphi(L3_vec)) * (h1 * self.cmb[typ].gradCl_spline(L3) - h2 * self.cmb[typ].gradCl_spline(L4))

    def normalisation_vector(self, typ, Ls, dL=2, Ntheta=100, curl=True):
        """

        Parameters
        ----------
        Ls
        dL
        Ntheta

        Returns
        -------

        """
        Lmax = 4000
        Lmin = 30
        self._initialisation_check(typ, Lmax)
        Ls3 = np.arange(Lmin, Lmax+1, dL)
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        for iii, L in enumerate(Ls):
            I = np.zeros(np.size(Ls3))
            for jjj, L3 in enumerate(Ls3):
                L3_vec = vector.obj(rho=L3, phi=thetas)
                L_vec =  vector.obj(rho=L, phi=0)
                g = self.weight_function_vector(typ, L3_vec, L_vec, curl)
                f = self.response_vector(typ, L3_vec, L_vec, curl)
                I[jjj] = 2 * dTheta *np.sum(L3 * g * f)
            A[iii] = InterpolatedUnivariateSpline(Ls3,I).integral(Lmin,Lmax)/((2*np.pi)**2)
        return A

    def weight_function(self, typ, L1, L2, theta, curl):
        L3 = self._get_third_L(L1,L2,theta)
        w = np.ones(np.shape(L3))
        w[L3<3] = 0
        w[L3 > 6000] = 0
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        denom = 2*(self.cmb[typ1].Cl_spline(L1)+self.cmb[typ1].N_spline(L1))*(self.cmb[typ2].Cl_spline(L3)+self.cmb[typ2].N_spline(L3))
        return w*self.response(typ, L1, L2, theta, curl)/denom

    def _initialise(self, typ, Lmax=6000):
        if self.cmb[typ].initialised:
            return
        self.cmb[typ] = self.CMBsplines()
        Cl_lens = self._cosmo.get_lens_ps(typ, Lmax)
        Ls = np.arange(np.size(Cl_lens))
        Cl_lens_spline = InterpolatedUnivariateSpline(Ls[1:], Cl_lens[1:])
        self.cmb[typ].Cl_spline = Cl_lens_spline
        gradCl_lens = self._cosmo.get_grad_lens_ps(typ, Lmax)
        gradCl_lens_spline = InterpolatedUnivariateSpline(np.arange(np.size(gradCl_lens))[1:], gradCl_lens[1:])
        self.cmb[typ].gradCl_spline = gradCl_lens_spline
        N = self._noise.get_cmb_gaussian_N(typ, ellmax=Lmax)
        N_spline = InterpolatedUnivariateSpline(np.arange(np.size(N))[1:], N[1:])
        self.cmb[typ].N_spline = N_spline
        self.cmb[typ].initialised = True
        self._initialise(typ[::-1], Lmax)

    def initialise(self, Lmax=6000):
        typs = np.char.array(list("TEB"))
        C = np.triu(typs[:, None] + typs[None, :]).flatten()
        args = C[C!='']
        for arg in args:
            self._initialise(arg,Lmax)

    def gmv_normalisation(self, Ls, dL=10, Ntheta=100, curl=True, explicit=False):
        typs = np.char.array(list("TEB"))
        Lmax = 4000
        Lmin = 30
        L_prims = np.arange(Lmin, Lmax + 1, dL)[None, :]
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)[:, None]
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        for iii, L in enumerate(Ls):
            I = 0
            for i in typs:
                for j in typs:
                    g_ij = self.gmv_weight_function(i+j, L_prims,L,thetas, curl, explicit)
                    # f_ji = self.response(j + i, L_prims,L,thetas, curl)
                    # I += 2 * dTheta * dL * np.sum(L_prims * g_ij * f_ji)
                    f_ij = self.response(i+j, L_prims,L,thetas, curl)
                    I += 2 * dTheta * dL * np.sum(L_prims * g_ij * f_ij)
            A[iii] = I / ((2 * np.pi) ** 2)
        return A/2

    def normalisation(self, typ, Ls, dL=2, Ntheta=100, curl=True):
        """

        Parameters
        ----------
        Ls
        dL
        Ntheta

        Returns
        -------

        """
        Lmax = 4000
        Lmin = 30
        self._initialisation_check(typ, Lmax)
        L_prims = np.arange(Lmin, Lmax+1, dL)[None, :]
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)[:,None]
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        for iii, L in enumerate(Ls):
            g = self.weight_function(typ, L_prims, L, thetas, curl)
            f = self.response(typ, L_prims, L, thetas, curl)
            I = 2 * dTheta * dL *np.sum(L_prims * g * f)
            A[iii] = I/((2*np.pi)**2)
        return A

    def normalisation2(self, typ, Ls, dL=2, Ntheta=100, curl=True):
        """

        Parameters
        ----------
        Ls
        dL
        Ntheta

        Returns
        -------

        """
        Lmax = 4000
        Lmin = 30
        self._initialisation_check(typ, Lmax)
        L_prims = np.arange(Lmin, Lmax+1, dL)
        dTheta = np.pi / Ntheta
        thetas = np.arange(dTheta, np.pi + dTheta, dTheta, dtype=float)
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1)*Ls
        for iii, L in enumerate(Ls):
            I = np.zeros(np.size(L_prims))
            for jjj, L_prim in enumerate(L_prims):
                g = self.weight_function(typ, L_prim, L, thetas, curl)
                f = self.response(typ, L_prim, L, thetas, curl)
                I[jjj] = 2 * dTheta *np.sum(L_prim * g * f)
            A[iii] = InterpolatedUnivariateSpline(L_prims,I).integral(Lmin,Lmax)/((2*np.pi)**2)
        return A

if __name__ == '__main__':
    import time
    qe = QE()
    # samp1 = np.arange(30, 40, 5)
    # samp2 = np.logspace(1, 3, 10) * 4
    # Ls = np.concatenate((samp1, samp2))
    # print((Ls*(Ls+1))**2/qe.normalisation("TT",Ls,dL=2,Ntheta=100)/4)
    # print((Ls * (Ls + 1))**2 / qe.normalisation2("TT", Ls, dL=2, Ntheta=100) / 4)
    # import matplotlib.pyplot as plt
    # plt.plot(Ls, (Ls * (Ls + 1))**2 / qe.gmv_normalisation(Ls, dL=2, Ntheta=100) / 4)
    # plt.show()
    t1 = time.time()
    L=100
    print(L)
    norm = qe.gmv_normalisation(L, 2, 100, True, True)
    print(norm)
    print(L ** 4 / norm / 4)
    norm = qe.gmv_normalisation(L, 2, 100, True, False)
    print(norm)
    print(L**4/norm/4)
    print("----------")
    L = 1000
    print(L)
    norm = qe.gmv_normalisation(L, 2, 100, True, True)
    print(norm)
    print(L ** 4 / norm / 4)
    norm = qe.gmv_normalisation(L, 2, 100, True, False)
    print(norm)
    print(L ** 4 / norm / 4)
    print("----------")
    L = 750
    print(L)
    norm = qe.gmv_normalisation(L, 2, 100, True, True)
    print(norm)
    print(L ** 4 / norm / 4)
    norm = qe.gmv_normalisation(L, 2, 100, True, False)
    print(norm)
    print(L ** 4 / norm / 4)
    print("----------")
    print("TT")
    weight = qe.gmv_weight_function("TT", 1000, 1500, 3, True, True)
    print(weight)
    weight = qe.gmv_weight_function("TT", 1000, 1500, 3, True, False)
    print(weight)
    print("----------------")
    print("TE")
    weight = qe.gmv_weight_function("TE", 1000, 2500, 1.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("TE", 1000, 2500, 1.5, True, False)
    print(weight)
    print("----------------")
    print("ET")
    weight = qe.gmv_weight_function("ET", 1000, 1500, 1.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("ET", 1000, 1500, 1.5, True, False)
    print(weight)
    print("----------------")
    print("EE")
    weight = qe.gmv_weight_function("EE", 1000, 1500, 0.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("EE", 1000, 1500, 0.5, True, False)
    print(weight)
    print("----------------")
    print("EB")
    weight = qe.gmv_weight_function("EB", 980, 1530, 2.9, True, True)
    print(weight)
    weight = qe.gmv_weight_function("EB", 980, 1530, 2.9, True, False)
    print(weight)
    print("----------------")
    print("BE")
    weight = qe.gmv_weight_function("BE", 1000, 1500, 1.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("BE", 1000, 1500, 1.5, True, False)
    print(weight)
    print("----------------")
    print("BT")
    weight = qe.gmv_weight_function("BT", 1000, 1500, 1.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("BT", 1000, 1500, 1.5, True, False)
    print(weight)
    print("----------------")
    print("TB")
    weight = qe.gmv_weight_function("TB", 1000, 1500, 1.5, True, True)
    print(weight)
    weight = qe.gmv_weight_function("TB", 1000, 1500, 1.5, True, False)
    print(weight)
    print("----------------")
    t2 = time.time()

