from bispectra import Bispectra
from fisher import Fisher
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class Bias:


    class F_L_holder:


        def __init__(self, spline, typs, nu):
            self.spline = spline
            self.typs = typs
            self.nu = nu



    def __init__(self, N0_file, N0_offset=0, N0_ell_factors=True):
        self.bi = Bispectra()
        self._fisher = Fisher(N0_file, N0_offset, N0_ell_factors)
        self._f_L = None

    def _build_F_L(self, typs, nu):
        Ls1 = np.arange(30, 40, 2)
        Ls2 = np.logspace(1, 3, 100) * 4
        sample_Ls = np.concatenate((Ls1, Ls2))
        _, F_L = self._fisher.get_F_L(typs, Ls=sample_Ls, Ntheta=100, nu=nu)
        self._F_L = self.F_L_holder(InterpolatedUnivariateSpline(sample_Ls, F_L), typs, nu)

    def _get_cov_invs(self, typ, typs, C_inv):
        combo1_idx1 = np.where(typs == typ[0])[0][0]
        combo1_idx2 = np.where(typs == typ[2])[0][0]
        combo2_idx1 = np.where(typs == typ[1])[0][0]
        combo2_idx2 = np.where(typs == typ[3])[0][0]

        cov_inv1 = C_inv[combo1_idx1][combo1_idx2]
        cov_inv2 = C_inv[combo2_idx1][combo2_idx2]
        return cov_inv1, cov_inv2

    def _mixed_bi_element(self, typ, typs, L, L1, L2, nu, C_inv):
        bi_typ = typ[:2] + "w"
        p = typ[2]
        q = typ[3]
        bi = self.bi.get_bispectrum(bi_typ, L1, L2, L, M_spline=True, nu=nu)
        cov_inv1, cov_inv2 = self._get_cov_invs(typ, typs, C_inv)
        Cl_pk = self._fisher.get_Cov(p+"k", ellmax=4000, nu=nu)
        Cl_qk = self._fisher.get_Cov(q+"k", ellmax=4000, nu=nu)
        return bi * cov_inv1[L1] * cov_inv2[L2] * (Cl_pk[L1]*Cl_qk[L2] + Cl_pk[L2]*Cl_qk[L1])


    def _mixed_bispectrum(self, typs, L, L1, L2, nu):
        F_L = self._F_L.spline(L)
        typs = np.char.array(typs)
        C_inv = self._fisher.get_C_inv(typs, Lmax=4000, nu=nu)
        all_combos = typs[:, None] + typs[None, :]
        combos = all_combos.flatten()
        Ncombos = np.size(combos)
        perms = 0
        mixed_bi = None
        for iii in np.arange(Ncombos):
            for jjj in np.arange(iii, Ncombos):
                typ = combos[iii] + combos[jjj]
                mixed_bi_element = self._mixed_bi_element(typ, typs, L, L1, L2, nu, C_inv)
                if combos[iii] != combos[jjj]:
                    factor = 2
                else:
                    factor = 1
                perms += factor
                if mixed_bi is None:
                    mixed_bi = mixed_bi_element
                else:
                    mixed_bi += factor * mixed_bi_element
        if perms != np.size(typs) ** 4:
            raise ValueError(f"{perms} permutations computed, should be {np.size(typs) ** 4}")
        return 2/(F_L * L1**2 * L2**2) * mixed_bi

    def mixed_bispectrum(self, typs, L, L1, L2, nu=353e9):
        typs = list(typs)
        if self._F_L is None or self._F_L.typs != typs or self._F_L.nu != nu:
            self._build_F_L(typs, nu)
        return self._mixed_bispectrum(typs, L, L1, L2, nu)
