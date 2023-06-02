import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import omegaqe
from omegaqe.noise import Noise
from sympy.matrices import Matrix
from sympy import lambdify
import vector


class QE:

    class CMBsplines:

        def __init__(self):
            self.initialised = False
            self.lenCl_spline = None
            self.gradCl_spline = None
            self.N_spline = None

    def __init__(self, exp, deltaT=None, beam=None, init=True, fields="TEB", L_cuts=(None,None,None,None), data_dir=omegaqe.DATA_DIR):
        self._noise = Noise()
        self.cosmo = self._noise.cosmo
        self.cmb = dict.fromkeys(self._cmb_types(), self.CMBsplines())
        if init:
            self.initialise(exp, deltaT, beam, fields=fields, data_dir=data_dir)
        else:
            self._cov_inv_fields = "uninitialised"
        self._setup_qe_L_cuts(L_cuts)

    def _setup_qe_L_cuts(self, L_cuts):
        self.T_Lmin = L_cuts[0]
        self.T_Lmax = L_cuts[1]
        self.P_Lmin = L_cuts[2]
        self.P_Lmax = L_cuts[3]


    def _cmb_types(self):
        types = np.char.array(list("TEB"))
        return (types[:, None] + types[None, :]).flatten()

    def _get_cmb_cl(self, ells, fields, typ):
        if typ == "lensed":
            return self.cmb[fields].lenCl_spline(ells)
        elif typ == "gradient":
            return self.cmb[fields].gradCl_spline(ells)

    def parse_fields(self, fields="TEB", unique=False, includeBB=False):
        typs = np.char.array(list(fields))
        typs_matrix = typs[:, None] + typs[None, :]
        if unique:
            XYs = np.triu(typs_matrix).flatten()
            XYs = XYs[XYs != '']
        else:
            XYs = typs_matrix.flatten()
        if includeBB:
            return XYs
        return XYs[XYs != "BB"]

    def _initialisation_check(self):
        if self._cov_inv_fields == "uninitialised":
            raise ValueError("QE class uninitialised, first call 'initialise'.")

    def get_log_sample_Ls(self, Lmin, Lmax, Nells=500, dL_small=1):
        floaty = Lmax / 1000
        samp1 = np.arange(Lmin, floaty * 10, dL_small)
        samp2 = np.logspace(1, 3, Nells-np.size(samp1)) * floaty
        return np.concatenate((samp1, samp2))

    def get_Lmin_Lmax(self, fields, gmv, strict=True):
        T_str = "T" if gmv else "TT"
        if fields == T_str:
            return self.T_Lmin, self.T_Lmax
        if "T" not in fields:
            return self.P_Lmin, self.P_Lmax
        if strict:
            return np.max((self.T_Lmin, self.P_Lmin)), np.min((self.T_Lmax, self.P_Lmax))
        return np.min((self.T_Lmin, self.P_Lmin)), np.max((self.T_Lmax, self.P_Lmax))

    def _get_L_cut_weights(self, typ, L1, L2):
        Lmin, Lmax = self.get_Lmin_Lmax(typ, False, strict=True)
        w1 = np.ones(np.shape(L1))
        w1[L1 < Lmin] = 0
        w1[L1 > Lmax] = 0
        w2 = np.ones(np.shape(L2))
        w2[L2 < Lmin] = 0
        w2[L2 > Lmax] = 0
        return w1, w2

    def gmv_normalisation(self, Ls, curl, fields="TEB", resp_ps="gradient", T_Lmin=30, T_Lmax=3000, P_Lmin=30, P_Lmax=5000):
        """
        Parameters
        ----------
        Ls
        curl
        Returns
        -------
        """
        self._initialisation_check()
        self.T_Lmin = T_Lmin
        self.T_Lmax = T_Lmax
        self.P_Lmin = P_Lmin
        self.P_Lmax = P_Lmax
        Lmin, Lmax = self.get_Lmin_Lmax(fields, True, strict=False)
        ells = self.get_log_sample_Ls(Lmin, Lmax, Nells=500)
        Ntheta = 300
        thetas = np.linspace(0, np.pi, Ntheta)
        I1 = np.zeros(np.size(ells))
        XYs = self.parse_fields(fields, unique=False)
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
                I2 = 0
                for XY in XYs:
                    w = np.ones(np.size(Ls3))
                    resp = self.response(XY, L_vec, ell_vec, curl, resp_ps)
                    g = self.gmv_weight_function(XY, L_vec, ell_vec, curl, fields, resp_ps, apply_Lcuts=True)
                    I2 += w * resp * g
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(Lmin, Lmax) / ((2 * np.pi) ** 2)
        return 1 / A

    def normalisation(self, typ, Ls, curl, resp_ps="gradient", T_Lmin=30, T_Lmax=3000, P_Lmin=30, P_Lmax=5000):
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
            return self.gmv_normalisation(Ls, curl, resp_ps=resp_ps, T_Lmin=T_Lmin, T_Lmax=T_Lmax, P_Lmin=P_Lmin, P_Lmax=P_Lmax)
        self._initialisation_check()
        self.T_Lmin = T_Lmin
        self.T_Lmax = T_Lmax
        self.P_Lmin = P_Lmin
        self.P_Lmax = P_Lmax
        Lmin, Lmax = self.get_Lmin_Lmax(typ, False, strict=False)
        ells = self.get_log_sample_Ls(Lmin, Lmax)
        Ntheta = 300
        thetas = np.linspace(0, np.pi, Ntheta)
        I1 = np.zeros(np.size(ells))
        N_Ls = np.size(Ls)
        A = np.zeros(N_Ls)
        if N_Ls == 1:
            Ls = np.ones(1) * Ls
        for iii, L in enumerate(Ls):
            L_vec = vector.obj(rho=L, phi=0)
            for jjj, ell in enumerate(ells):
                ell_vec = vector.obj(rho=ell, phi=thetas)
                resp = self.response(typ, L_vec, ell_vec, curl, resp_ps)
                g = self.weight_function(typ, L_vec, ell_vec, curl, gmv=False, resp_ps=resp_ps, apply_Lcuts=True)
                I2 = g * resp
                I1[jjj] = 2 * ell * InterpolatedUnivariateSpline(thetas, I2).integral(0, np.pi)
            A[iii] = InterpolatedUnivariateSpline(ells, I1).integral(Lmin, Lmax) / ((2 * np.pi) ** 2)
        return 1 / A

    def gmv_weight_function(self, typ, L_vec, ell_vec, curl, fields="TEB", resp_ps="gradient", apply_Lcuts=False):
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
        self._initialisation_check()
        weight = 0
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        p = typ[0]
        q = typ[1]
        XYs = self.parse_fields(fields, unique=False)
        for ij in XYs:
            i = ij[0]
            j = ij[1]
            C_inv_ip = self._get_cmb_Cov_inv_spline(i + p, fields)(ell)
            C_inv_jq = self._get_cmb_Cov_inv_spline(j + q, fields)(L3)
            weight_tmp = self.response(i + j, L_vec, ell_vec, curl, resp_ps) * C_inv_ip * C_inv_jq
            if apply_Lcuts:
                w1, w2 = self._get_L_cut_weights(typ, ell, L3)
                weight_tmp *= w1 * w2
            weight += weight_tmp
        return weight / 2


    def weight_function(self, typ, L_vec, ell_vec, curl, gmv=False, fields="TEB", resp_ps="gradient", apply_Lcuts=False):
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
        # TODO: all weight funcs (except TT) are possibly wrong
        if gmv:
            return self.gmv_weight_function(typ, L_vec, ell_vec, curl, fields, resp_ps, apply_Lcuts=apply_Lcuts)
        self._initialisation_check()
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        typ1 = typ[0]+typ[0]
        typ2 = typ[1]+typ[1]
        C_typ1 = self._get_cmb_cov(typ1, ell)
        C_typ2 = self._get_cmb_cov(typ2, L3)
        fac = 0.5 if typ1 == typ2 else 1
        if apply_Lcuts:
            w1, w2 = self._get_L_cut_weights(typ, ell, L3)
            return w1*w2*fac*self.response(typ, L_vec, ell_vec, curl, resp_ps)/(C_typ1 * C_typ2)
        return fac*self.response(typ, L_vec, ell_vec, curl, resp_ps)/(C_typ1 * C_typ2)

    def _response_phi(self, typ, L_vec, ell_vec, cl="gradient"):
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        theta12 = ell_vec.deltaphi(L3_vec)
        h1 = self._get_response_geo_fac(typ, 1, theta12=theta12)
        h2 = self._get_response_geo_fac(typ, 2, theta12=theta12)
        if typ == "BT" or typ == "TB" or typ == "TE" or typ == "ET":
            typ1 = typ2 = "TE"
        else:
            typ1 = typ[0] + typ[0]
            typ2 = typ[1] + typ[1]
        return ((L_vec @ ell_vec) * h1 * self._get_cmb_cl(ell, typ1, cl)) + ((L_vec @ L3_vec) * h2 * self._get_cmb_cl(L3, typ2, cl))

    def response(self, typ, L_vec, ell_vec, curl=True, cl="gradient"):
        """

        Parameters
        ----------
        typ
        L_vec
        ell_vec
        curl
        cl

        Returns
        -------

        """
        # TODO: also the curl terms (i.e. C^TP and C^PP) are not implimented
        if not curl:
            return self._response_phi(typ, L_vec, ell_vec, cl)
        L = L_vec.rho
        ell = ell_vec.rho
        L3_vec = L_vec - ell_vec
        L3 = L3_vec.rho
        theta12 = ell_vec.deltaphi(L3_vec)
        h1 = self._get_response_geo_fac(typ, 1, theta12=theta12)
        h2 = self._get_response_geo_fac(typ, 2, theta12=theta12)
        if typ == "BT" or typ == "TB" or typ == "TE" or typ == "ET":
            typ1 = typ2 = "TE"
        else:
            typ1 = typ[0] + typ[0]
            typ2 = typ[1] + typ[1]
        return L * ell * np.sin(ell_vec.deltaphi(L_vec)) * (h1 * self._get_cmb_cl(ell, typ1, cl) - h2 * self._get_cmb_cl(L3, typ2, cl))

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

    def _get_cmb_cov(self, typ, Ls):
        return self.cmb[typ].lenCl_spline(Ls) + self.cmb[typ].N_spline(Ls)

    def _cmb_Cov_inv(self, fields):
        typs = np.char.array(list(fields))
        C = typs[:, None] + typs[None, :]
        args = C.flatten()
        C_sym = Matrix(C)
        C_inv = C_sym.inv()
        C_inv_func = lambdify(args, C_inv)
        Lmin = 2
        Lmax = 5000
        Ls = np.arange(Lmin, Lmax + 1)
        Covs = [self._get_cmb_cov(arg, Ls) for arg in args]
        return Ls, C_inv_func(*Covs)

    def _build_cmb_Cov_inv_splines(self, fields):
        self._cov_inv_fields = fields
        N_fields = len(self._cov_inv_fields)
        Ls, C_inv = self._cmb_Cov_inv(fields)
        C_inv_splines = np.empty((N_fields,N_fields), dtype=InterpolatedUnivariateSpline)
        for iii in range(N_fields):
            for jjj in range(N_fields):
                C_inv_ij = C_inv[iii,jjj]
                C_inv_splines[iii, jjj] = InterpolatedUnivariateSpline(Ls[2:], C_inv_ij[2:])
        self.C_inv_splines = C_inv_splines


    def _get_cmb_Cov_inv_spline(self, typ, fields):
        if fields != self._cov_inv_fields:
            self._build_cmb_Cov_inv_splines(fields)
        typs = np.char.array(list(fields))

        idx1 = np.where(typs == typ[0])[0][0]
        idx2 = np.where(typs == typ[1])[0][0]

        cov_inv = self.C_inv_splines[idx1][idx2]
        return cov_inv



    def _initialise(self, typ, deltaT, beam, exp="SO", data_dir=omegaqe.DATA_DIR):
        if self.cmb[typ].initialised:
            return
        self.cmb[typ] = self.CMBsplines()
        Cl_lens = self.cosmo.get_lens_ps(typ, 6000)
        Ls = np.arange(np.size(Cl_lens))
        Cl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], Cl_lens[2:])
        self.cmb[typ].lenCl_spline = Cl_lens_spline

        gradCl_lens = self.cosmo.get_grad_lens_ps(typ, 6000)
        Ls = np.arange(np.size(gradCl_lens))
        gradCl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], gradCl_lens[2:])
        self.cmb[typ].gradCl_spline = gradCl_lens_spline

        N = self._noise.get_cmb_gaussian_N(typ, ellmax=6000, deltaT=deltaT, beam=beam, exp=exp, data_dir=data_dir)
        N_spline = InterpolatedUnivariateSpline(np.arange(np.size(N))[2:], N[2:])
        self.cmb[typ].N_spline = N_spline
        self.cmb[typ].initialised = True
        self._initialise(typ[::-1], deltaT, beam, exp)

    def initialise(self, exp="SO", deltaT=None, beam=None, fields="TEB", data_dir=omegaqe.DATA_DIR):
        """
        Parameters
        ----------
        Returns
        -------
        """
        if deltaT is None or beam is None:
            deltaT, beam = self._noise.get_noise_args(exp)
        args = self.parse_fields(fields, includeBB=True)
        for arg in args:
            self._initialise(arg, deltaT, beam, exp, data_dir)
        self._build_cmb_Cov_inv_splines(fields=fields)


    def _initialise_manual(self, typ, Cl_lens, gradCl_lens, N):
        if self.cmb[typ].initialised:
            return
        self.cmb[typ] = self.CMBsplines()
        Ls = np.arange(np.size(Cl_lens))
        Cl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], Cl_lens[2:])
        self.cmb[typ].lenCl_spline = Cl_lens_spline

        Ls = np.arange(np.size(gradCl_lens))
        gradCl_lens_spline = InterpolatedUnivariateSpline(Ls[2:], gradCl_lens[2:])
        self.cmb[typ].gradCl_spline = gradCl_lens_spline

        N_spline = InterpolatedUnivariateSpline(np.arange(np.size(N))[2:], N[2:])
        self.cmb[typ].N_spline = N_spline
        self.cmb[typ].initialised = True
        self._initialise_manual(typ[::-1], Cl_lens, gradCl_lens, N)

    def initialise_manual(self, typ, Cl_lens, gradCl_lens, N):
        """
        Parameters
        ----------
        typ
        Cl_lens
        gradCl_lens
        N
        Returns
        -------
        """
        self._initialise_manual(typ, Cl_lens, gradCl_lens, N)


if __name__ == '__main__': pass