import numpy as np
from cosmology import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline
from noise import Noise
from sympy.matrices import Matrix
from sympy import lambdify


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

    def response(self, typ, L1, L2, theta):
        L3 = self._get_third_L(L1,L2,theta)
        w = np.ones(np.shape(L3))
        w[L3 < 30] = 0
        theta3 = np.arcsin(L1*np.sin(theta)/L3)
        h1 = self._get_response_geo_fac(typ, 1, theta, theta3)
        h2 = self._get_response_geo_fac(typ, 2, theta, theta3)
        self._initialisation_check(typ)
        if typ == "BT" or typ == "TB":
            return w * L1 * L2 * np.sin(theta) * (h1 * self.cmb["TE"].gradCl_spline(L1) - h2 * self.cmb["TE"].gradCl_spline(L3))
        elif typ == "EB":
            return w * L1 * L2 * np.sin(theta) * (h1 * self.cmb["EE"].gradCl_spline(L1) - h2 * self.cmb["BB"].gradCl_spline(L3))
        elif typ == "BE":
            return w * L1 * L2 * np.sin(theta) * (h1 * self.cmb["BB"].gradCl_spline(L1) - h2 * self.cmb["EE"].gradCl_spline(L3))
        return w*L1*L2*np.sin(theta)*(h1*self.cmb[typ].gradCl_spline(L1) - h2*self.cmb[typ].gradCl_spline(L3))

    def geo_fac(self, typ, theta1=None, theta2=None, theta12=None):
        if typ == "T":
            return self._geo_fac_T()
        elif typ == "E":
            if theta12 is None:
                return self._geo_fac_E(theta1,theta2)
            return np.cos(2*theta12)
        elif typ == "B":
            if theta12 is None:
                return self._geo_fac_B(theta1,theta2)
            return np.sin(2*theta12)

    def _geo_fac_T(self):
        return 1

    def _geo_fac_E(self, theta1, theta2):
        return np.cos(2*(theta1 - theta2))

    def _geo_fac_B(self, theta1, theta2):
        return np.sin(2*(theta1-theta2))

    def _get_response_geo_fac(self, typ, num, theta1, theta2):
        if typ == "TT":
            return self._geo_fac_T()
        elif typ == "EE" or typ == "BB":
            return self._geo_fac_E(theta1,theta2)
        elif typ == "EB" or typ == "BE":
            return self._geo_fac_B(theta1,theta2)
        elif typ == "TE":
            if num == 1:
                return self._geo_fac_E(theta1,theta2)
            return self._geo_fac_T()
        elif typ == "ET":
            if num == 1:
                return self._geo_fac_T()
            return self._geo_fac_E(theta1,theta2)
        elif typ == "TB":
            if num == 1:
                return self._geo_fac_B(theta1,theta2)
            return 0
        elif typ == "BT":
            if num == 1:
                return 0
            return self._geo_fac_B(theta1,theta2)
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

    def _gmv_weight_numer(self, typ, L1, L2, theta, C_TT, C_EE, C_TE):
        L3 = self._get_third_L(L1, L2, theta)
        f_TT = self.response("TT", L1, L2, theta)
        f_TE = self.response("TE", L1, L2, theta)
        f_ET = self.response("ET", L1, L2, theta)
        f_EE = self.response("EE", L1, L2, theta)
        f_TB = self.response("TB", L1, L2, theta)
        f_EB = self.response("EB", L1, L2, theta)
        f_BT = self.response("BT", L1, L2, theta)
        f_BE = self.response("BE", L1, L2, theta)
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

    def gmv_weight_function(self, typ, L1, L2, theta, explicit=True):
        if explicit:
            C_TT = self._get_cmb_cov_spline("TT")
            C_EE = self._get_cmb_cov_spline("EE")
            C_TE = self._get_cmb_cov_spline("TE")
            denom = self._gmv_weight_denom(typ, L1, L2, theta, C_TT, C_EE, C_TE)
            numer = self._gmv_weight_numer(typ, L1, L2, theta, C_TT, C_EE, C_TE)
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
            weight += self.response(arg, L1, L2, theta)*C_inv_ip_spline(L1)*C_inv_jq_spline(L3)
        return weight/2

    def weight_function(self, typ, L1, L2, theta):
        L3 = self._get_third_L(L1,L2,theta)
        w = np.ones(np.shape(L3))
        w[L3<3] = 0
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        denom = 2*(self.cmb[typ1].Cl_spline(L1)+self.cmb[typ1].N_spline(L1))*(self.cmb[typ2].Cl_spline(L3)+self.cmb[typ2].N_spline(L3))
        return w*self.response(typ, L1, L2, theta)/denom

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

    def gmv_normalisation(self, Ls, dL=10, Ntheta=100):
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
                    g_ij = self.gmv_weight_function(i+j, L_prims,L,thetas)
                    f_ji = self.response(j + i, L_prims,L,thetas)
                    I += 2 * dTheta * dL * np.sum(L_prims * g_ij * f_ji)
            A[iii] = I / ((2 * np.pi) ** 2)
        return A/2

    def normalisation(self, typ, Ls, dL=10, Ntheta=100):
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
            g = self.weight_function(typ, L_prims, L, thetas)
            f = self.response(typ, L_prims, L, thetas)
            I = 2 * dTheta * dL *np.sum(L_prims * g * f)
            A[iii] = I/((2*np.pi)**2)
        return A

if __name__ == '__main__':
    import time
    qe = QE()
    Ls = np.arange(30,4000,100)
    # print(qe.gmv_normalisation(Ls))
    t1 = time.time()
    print("TT")
    weight = qe.gmv_weight_function("TT", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("TT", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("TE")
    weight = qe.gmv_weight_function("TE", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("TE", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("ET")
    weight = qe.gmv_weight_function("ET", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("ET", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("EE")
    weight = qe.gmv_weight_function("EE", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("EE", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("EB")
    weight = qe.gmv_weight_function("EB", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("EB", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("BE")
    weight = qe.gmv_weight_function("BE", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("BE", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("BT")
    weight = qe.gmv_weight_function("BT", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("BT", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    print("TB")
    weight = qe.gmv_weight_function("TB", 100, 150, 1.5)
    print(weight)
    weight = qe.gmv_weight_function("TB", 100, 150, 1.5, False)
    print(weight)
    print("----------------")
    t2 = time.time()

