import os
from os.path import join as opj
import numpy as np
from lenspyx.utils_hp import alm_copy
from lenspyx.remapping.deflection import deflection
from plancklens.filt import filt_simple
from plancklens.qcinv import cd_solve
from plancklens import utils
from plancklens.filt import filt_cinv
from plancklens import qest, qresp, nhl
from plancklens.n1 import n1
from plancklens.sims import phas, maps
from DEMNUnii.demnunii import Demnunii
from omegaqe.noise import Noise
import omegaqe.postborn as postborn
import shutil
from delensalot.core.opfilt.MAP_opfilt_iso_p import alm_filter_nlev_wl as alm_filter_nlev_wl_pol_only
from delensalot.core.opfilt.MAP_opfilt_iso_tp import alm_filter_nlev_wl as alm_filter_nlev_wl_teb
from delensalot.core.opfilt.MAP_opfilt_iso_t import alm_filter_nlev_wl as alm_filter_nlev_wl_t_only
from delensalot.utility import utils_steps
from delensalot.core.iterator.cs_iterator_multi import iterator_cstmf
from delensalot.core.iterator.statics import rec as Rec
from lensitbiases import n1_fft
from scipy.interpolate import InterpolatedUnivariateSpline
import copy

class Reconstruction:

    class MyMapLib:

        def __init__(self, lmax_map, maps_filename, sht):
            self.lmax = lmax_map
            self.sht = sht
            self.maps_filename = maps_filename
            self.maps = self.sht.read_map(maps_filename)

        def get_sim_tlm(self, idx):
            return self.sht.map2alm(self.maps[0], self.lmax)

        def _get_map_eblm(self, idx):
            return self.sht.map2alm_spin(self.maps[1:], 2, self.lmax)

        def get_sim_elm(self, idx):
            return self._get_map_eblm(idx)[0]

        def get_sim_blm(self, idx):
            return self._get_map_eblm(idx)[1]

        def hashdict(self):
            return {'cmbs': self.maps_filename, 'noise': self.maps_filename, 'data': self.maps_filename}

    def __init__(self, exp, filename=None, L_cuts=(30, 3000, 30, 5000), sim=None, nthreads=1, iter=False, noise=True, gmv=False):
        self.nthreads = nthreads
        self.filename = filename
        self.setup_env(exp, iter, sim)
        self.exp = exp
        self.dm = Demnunii(nthreads)
        self.sht = self.dm.sht
        self.noise = Noise(cosmology=self.dm.cosmo, full_sky=True)
        self.power = self.dm.power
        self.L_cuts = L_cuts
        self.Lmax_map = self.dm.Lmax_map
        self.indices = ["tt", "ee", "bb"]
        self.cl_unl, self.cl_len, self.cl_grad = self.get_cmb_cls()
        self.noise_cls = self.get_noise_cls(exp)
        self._initialise()
        self.setup = False
        self.iter_rec_data = None
        self.noise = noise
        self.gmv = gmv
        if filename is not None:
            self.setup_reconstruction(self.filename, seed=sim, iter=iter, noise=noise, gmv=gmv)

    def _initialise(self):
        print("Initialising filters and libraries for Plancklens")
        transfers = self.get_transfers()
        self.transfer_dict = {'t': transfers[0], 'e': transfers[1], 'b': transfers[2]}
        filters = self.get_filters(self.indices, self.cl_len, self.noise_cls)
        self.filt_dict = {'t': filters[0], 'e': filters[1], 'b': filters[2]}
        self.n1_lib = n1.library_n1(os.path.join(self.temp, 'n1'), self.cl_grad['tt'], self.cl_grad['ee'], self.cl_grad['bb'], lmaxphi=5000)

    def get_transfers(self):
        transfers = [np.ones(self.Lmax_map + 1) for _ in self.indices]
        return transfers
    
    def get_Lmax_filt(self):
        return np.max(self.L_cuts)

    def _get_Lmin(self, typ):
        if typ == "tt":
            return self.L_cuts[0]
        if typ == "ee" or typ == "bb":
            return self.L_cuts[2]
        return np.max([self.L_cuts[0], self.L_cuts[2]])

    def _get_Lmax(self, typ):
        if typ == "tt":
            return self.L_cuts[1]
        if typ == "ee" or typ == "bb":
            return self.L_cuts[3]
        return np.min([self.L_cuts[1], self.L_cuts[3]])

    def setup_env(self, exp, iter, sim):
        if not 'PLENS' in os.environ.keys():
            plancklens_cachedir = f"_tmp"
            os.environ['PLENS'] = plancklens_cachedir
            print(f"Setting up Plancklens cahe at {plancklens_cachedir}")
        self.temp = os.path.join(os.environ['PLENS'], 'demnunii', f"{exp}_{iter}", str(sim))
        if os.path.exists(self.temp) and os.path.isdir(self.temp):
            print(f"Removing existing plancklens cache at {self.temp}")
            shutil.rmtree(self.temp)

    def get_cmb_cls(self):
        Tcmb = 2.7255
        fac = (Tcmb * 1e6) ** 2
        indices = self.indices + ['te']
        cl_unl = {idx : self.dm.cosmo.get_unlens_ps(idx.upper()) * fac for idx in indices}
        cl_len = {idx : self.dm.cosmo.get_lens_ps(idx.upper()) * fac for idx in indices}
        cl_grad = {idx: self.dm.cosmo.get_grad_lens_ps(idx.upper()) * fac for idx in indices}
        return cl_unl, cl_len, cl_grad

    def get_noise_cls(self, exp):
        Tcmb = 2.7255
        fac = (Tcmb * 1e6) ** 2
        return {idx[0]: self.noise.get_cmb_gaussian_N(idx.upper(), None, None, ellmax=self.Lmax_map, exp=exp) * fac for idx in self.indices}

    def _apply_cuts(self, filts, indices):
        for iii, filt in enumerate(filts):
            filt[:self._get_Lmin(indices[iii])] *= 0.
            Lmax = self._get_Lmax(indices[iii])
            if np.size(filt) > Lmax:
                filt[Lmax + 1:] *= 0.
        return filts

    def get_filters(self, indices, cl_dict, noise_cls=None):
        Lmax_filt = self.get_Lmax_filt()
        if Lmax_filt >= self.Lmax_map:
            Lmax_filt = self.Lmax_map - 1
        filts = [utils.cli(cl_dict[idx][:Lmax_filt + 1] + noise_cls[idx[0]][:Lmax_filt + 1]) if noise_cls is not None else
                utils.cli(cl_dict[idx][:Lmax_filt + 1]) for idx in indices]
        return self._apply_cuts(filts, indices)

    def _raise_typ_error(self, typ):
        raise ValueError(f"QE type {typ} not recognized. Recognized types included TEB (for mv), EB (for pol only), or T.")

    def _get_qe_key(self, typ, curl=False):
        potential = "x" if curl else "p"
        if typ == "TEB":
            return potential
        if typ == "T":
            return potential + "tt"
        if typ == "EB":
            return potential + "_p"
        self._raise_typ_error(typ)

    def _get_cov(self, cl_dict, iii, jjj, noise):
        idx_i = self.indices[iii][0]
        idx_j = self.indices[jjj][0]
        ij = idx_i + idx_j
        N = np.zeros(self.Lmax_map + 1)
        cl = np.zeros(self.Lmax_map + 1)
        if idx_i == idx_j:
            if noise:
                N = self.noise_cls[idx_i]
            cl = cl_dict[ij][:self.Lmax_map+1]
        elif ij == "te" or ij == "et":
            cl = cl_dict['te'][:self.Lmax_map + 1]
        return cl + N
        
    def _get_filt_matrix(self, cl_dict, noise=True):
        mat = np.zeros((self.Lmax_map + 1, 3, 3))
        for iii in np.arange(3):
            for jjj in np.arange(3):         
                mat[:, iii,jjj] = self._get_cov(cl_dict, iii, jjj, noise)
        c_inv = np.linalg.pinv(mat)
        fal_dict = {}
        for iii in np.arange(3):
            idx_i = self.indices[iii][0]
            for jjj in np.arange(3):
                idx_j = self.indices[jjj][0]
                c_inv_ij = c_inv[:,iii,jjj]
                Lmin = self._get_Lmin(idx_i+idx_j)
                Lmax = self._get_Lmax(idx_i+idx_j)
                c_inv_ij[:Lmin] *= 0.
                c_inv_ij[Lmax+1:] *= 0.
                fal_dict[idx_i+idx_j] = c_inv_ij
        return fal_dict

    def setup_reconstruction(self, TQUmaps_filename, seed=None, iter=False, noise=True, gmv=False):
        print(f"Setting up reconstruction for file: {TQUmaps_filename}")
        libdir_pixphas = os.path.join(self.temp, 'phas_lmax%s' % self.Lmax_map)
        rng_state = np.random.get_state
        if seed is not None:
            print(f"   Setting random_state for noise map generation with seed {seed}.")
            rng = np.random.RandomState(seed)
            rng_state = rng.get_state
        cl_wf = self.cl_len if iter else self.cl_grad
        cl_wf_nobb = copy.deepcopy(cl_wf)
        cl_wf_nobb['bb'] *=0.
        pix_phas = phas.lib_phas(libdir_pixphas, 3, self.Lmax_map, get_state_func=rng_state)
        noise_cls = self.noise_cls if noise else {idx[0]: np.zeros(np.size(self.noise_cls[idx[0]])) for idx in self.indices}
        self.maps_lib = maps.cmb_maps_harmonicspace(self.MyMapLib(self.Lmax_map, TQUmaps_filename, self.sht), self.transfer_dict, noise_cls, noise_phas=pix_phas)
        if gmv:
            filt_matrix = self._get_filt_matrix(self.cl_len, noise=True)
            weighted_maps_lib = filt_simple.library_fullsky_alms_jTP(os.path.join(self.temp, 'ivfs'), self.maps_lib, self.transfer_dict, cl_wf_nobb, filt_matrix)
            self.qlms_lib = qest.library_jtTP(os.path.join(self.temp, 'qlms_dd'), weighted_maps_lib, weighted_maps_lib, self.dm.nside, lmax_qlm=self.Lmax_map)
            self.qresp_lib = qresp.resp_lib_simple(os.path.join(self.temp, 'qresp'), self.Lmax_map, cl_wf_nobb, cl_wf, filt_matrix, self.Lmax_map)
        else:
            weighted_maps_lib = filt_simple.library_fullsky_alms_sepTP(os.path.join(self.temp, 'ivfs'), self.maps_lib, self.transfer_dict, cl_wf, self.filt_dict['t'], self.filt_dict['e'], self.filt_dict['b'])
            self.qlms_lib = qest.library_sepTP(os.path.join(self.temp, 'qlms_dd'), weighted_maps_lib, weighted_maps_lib, cl_wf['te'], self.dm.nside, lmax_qlm=self.Lmax_map)
            self.qresp_lib = qresp.resp_lib_simple(os.path.join(self.temp, 'qresp'), self.Lmax_map, cl_wf, cl_wf, self.filt_dict, self.Lmax_map)
        self.rdn0_lib = nhl.nhl_lib_simple(os.path.join(self.temp, 'rdn0'), weighted_maps_lib, cl_wf, self.Lmax_map)
        self.setup = True

    def get_itlib(self, typ, cg_tol:float, libdir_iterator, chain_descrs):
        """Return iterator instance for simulation idx and qe_key type k
            Args:
                typ: EB only accepted type for now
                cg_tol: tolerance of conjugate-gradient filter
        """
        simidx = -1
        assert typ in ["T", "EB", "TEB"], typ
        qe_key = self._get_qe_key(typ)

        plm0 = self.qlms_lib.get_sim_qlm('p' + qe_key[1:], simidx) 
        olm0 = self.qlms_lib.get_sim_qlm('x' + qe_key[1:], simidx) 

        Rpp, Roo = qresp.get_response(qe_key, self.Lmax_map, 'p', self.cl_len, self.cl_len,
                                            self.filt_dict, lmax_qlm=self.Lmax_map)[0:2]

        cpp = self._get_Cl_phi()[:self.Lmax_map + 1]
        coo = self._get_Cl_curl()[:self.Lmax_map + 1]

        lens_Lmin = 10
        cpp[:lens_Lmin] *= 0.
        coo[:lens_Lmin] *= 0.

        WF_p = cpp * utils.cli(cpp + utils.cli(Rpp))
        WF_o = coo * utils.cli(coo + utils.cli(Roo))

        plm0 = alm_copy(plm0, None, self.Lmax_map, self.Lmax_map)  
        plm0 = self.sht.almxfl(plm0, utils.cli(Rpp)) 
        plm0 = self.sht.almxfl(plm0, WF_p)  
        plm0 = self.sht.almxfl(plm0, cpp > 0)

        olm0 = alm_copy(olm0, None, self.Lmax_map, self.Lmax_map)  
        olm0 = self.sht.almxfl(olm0, utils.cli(Roo)) 
        olm0 = self.sht.almxfl(olm0, WF_o)  
        olm0 = self.sht.almxfl(olm0, coo > 0)

        filt_unl = self.get_filters(self.indices, self.cl_unl, self.noise_cls)
        filt_dict_unl = {'t': filt_unl[0], 'e': filt_unl[1], 'b': filt_unl[2]}
        Rpp_unl, Roo_unl = qresp.get_response(qe_key, self.Lmax_map, 'p', self.cl_unl, self.cl_unl,
                                            filt_dict_unl, lmax_qlm=self.Lmax_map)[0:2]


        ffi = deflection(self.sht.geom, np.zeros_like(plm0), self.Lmax_map, numthreads=self.nthreads, epsilon=1e-7)
        transfers =  self._apply_cuts(self.get_transfers(), self.indices)
        arcmin_to_rad = np.pi/180/60
        noise_t = np.sqrt(self.noise_cls['t']) / arcmin_to_rad
        noise_e = np.sqrt(self.noise_cls['e']) / arcmin_to_rad
        noise_b = np.sqrt(self.noise_cls['b']) / arcmin_to_rad
        if typ == "EB":
            filtr = alm_filter_nlev_wl_pol_only(nlev_p=noise_e, ffi=ffi, transf=transfers[1],
                unlalm_info=(self.Lmax_map, self.Lmax_map), lenalm_info=(self.Lmax_map, self.Lmax_map), transf_b=transfers[2], nlev_b=noise_b)
            elm, blm = self.maps_lib.get_sim_pmap(simidx)
            datmaps = np.array([alm_copy(elm, None, self.Lmax_map, self.Lmax_map), alm_copy(blm, None, self.Lmax_map, self.Lmax_map)])
            wflm0=lambda : alm_copy(self.rdn0_lib.ivfs.get_sim_emliklm(simidx), None, self.Lmax_map, self.Lmax_map)
        elif typ == "TEB":
            filtr = alm_filter_nlev_wl_teb(nlev_t=noise_t, nlev_p=noise_e, ffi=ffi, transf=transfers[0], unlalm_info=(self.Lmax_map, self.Lmax_map), lenalm_info=(self.Lmax_map, self.Lmax_map),
                                    transf_e=transfers[1], transf_b=transfers[2], nlev_b=noise_b)
            elm, blm = self.maps_lib.get_sim_pmap(simidx)
            tlm = self.maps_lib.get_sim_tmap(simidx)
            datmaps = np.array([alm_copy(tlm, None, self.Lmax_map, self.Lmax_map),alm_copy(elm, None, self.Lmax_map, self.Lmax_map),alm_copy(blm, None, self.Lmax_map, self.Lmax_map)])
            wflm0 = lambda: np.array([alm_copy(self.rdn0_lib.ivfs.get_sim_tmliklm(simidx), None, self.Lmax_map, self.Lmax_map),alm_copy(self.rdn0_lib.ivfs.get_sim_emliklm(simidx), None, self.Lmax_map, self.Lmax_map)])
        elif typ == "T":
            filtr = alm_filter_nlev_wl_t_only(nlev_t=noise_t, ffi=ffi, transf=transfers[0], unlalm_info=(self.Lmax_map, self.Lmax_map), lenalm_info=(self.Lmax_map, self.Lmax_map))
            tlm = self.maps_lib.get_sim_tmap(simidx)
            datmaps = alm_copy(tlm, None, self.Lmax_map, self.Lmax_map)
            wflm0 = lambda: alm_copy(self.rdn0_lib.ivfs.get_sim_tmliklm(simidx), None, self.Lmax_map, self.Lmax_map)
        k_geom = filtr.ffi.geom 
        stepper = utils_steps.harmonicbump(xa=400, xb=1500)
        mean_field = np.zeros(np.shape(plm0), dtype=complex)
        iterator = iterator_cstmf(libdir_iterator, 'p', [(self.Lmax_map, self.Lmax_map), (self.Lmax_map, self.Lmax_map)], datmaps,
                                [plm0, olm0], [mean_field, mean_field], [Rpp_unl, Roo_unl], [cpp, coo], ('p', 'x'), self.cl_unl, filtr, k_geom,
                                chain_descrs(self.Lmax_map, cg_tol), stepper,
                                wflm0=wflm0)
        return iterator

    def _check_setup(self):
        if not self.setup:
            raise ValueError("Need to call 'setup_reconstruction' first. No qlm_lib has been instantiated.")
        
    def _setup_iter_libdir(self, qe_key):
        TEMP =  opj(self.temp, "iter_lib")
        version = "mathew"
        libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        lib_dir_iterator = libdir_iterators(qe_key, -1, version) 
        if os.path.exists(TEMP) and os.path.isdir(TEMP):
            print(f"Removing existing delensalot cache at {TEMP}")
            shutil.rmtree(TEMP)
        if not os.path.exists(lib_dir_iterator):
            os.makedirs(lib_dir_iterator)
        print("Caching things in " + TEMP)
        print("iterator folder: " + lib_dir_iterator)
        return lib_dir_iterator
    
    def calc_iter_rec(self, typ="EB", itmax=15):
        tol_iter = 1e-5 if typ == "TEB" else 1e-7
        soltn_cond = True
        qe_key = self._get_qe_key(typ)
        lib_dir_iterator = self._setup_iter_libdir(qe_key)
        chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, self.dm.nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
        if itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < itmax:
            itlib = self.get_itlib(typ, 1., lib_dir_iterator, chain_descrs)
            for iii in range(itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter)
                print("****Iterator: setting solcond to %s ****"%soltn_cond)
                itlib.chain_descr  = chain_descrs(self.Lmax_map, tol_iter)
                itlib.soltn_cond   = soltn_cond
                print("doing iter " + str(iii))
                itlib.iterate(iii, 'p')
        
        iters = np.arange(itmax + 1)
        self.iter_rec_data = Rec.load_plms(lib_dir_iterator, iters)
        self.iter_typ = typ
        self.niters = itmax

    def get_phi_rec(self, typ):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=False)
        qlm = self.qlms_lib.get_sim_qlm(qe_key, -1)
        resp = self.get_response(typ)
        qnorm = utils.cli(resp)
        return self.sht.almxfl(qlm, qnorm)

    def get_curl_rec(self, typ):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=True)
        qlm = self.qlms_lib.get_sim_qlm(qe_key, -1)
        resp = self.get_response(typ, True)
        qnorm = utils.cli(resp)
        return self.sht.almxfl(qlm, qnorm)
    
    def get_phi_rec_iter(self, iter):
        if self.iter_rec_data is None:
            raise ValueError("No iterative reconstruction found, run calc_iter_rec first.")
        rec_lms =  self.iter_rec_data[iter]
        phi_lm = rec_lms[:len(rec_lms)//2]
        normalisation = 1/self.get_wiener("phi", self.iter_typ, iter=True)
        normalisation[0] = 0
        return self.sht.almxfl(phi_lm, normalisation)

    def get_curl_rec_iter(self, iter):
        if self.iter_rec_data is None:
            raise ValueError("No iterative reconstruction found, run calc_iter_rec first.")
        rec_lms =  self.iter_rec_data[iter]
        curl_lm = rec_lms[len(rec_lms)//2:]
        normalisation = 1/self.get_wiener("curl", self.iter_typ, iter=True)
        normalisation[0] = 0
        return self.sht.almxfl(curl_lm, normalisation)
    
    def get_wiener(self, typ, qe_typ, iter):
        Cl = self._get_Cl_curl() if typ == "curl" else self._get_Cl_phi()
        N0 = self.noise.get_N0(typ, self.Lmax_map, qe=qe_typ, gmv=True, exp=self.exp, T_Lmin=self.L_cuts[0], T_Lmax=self.L_cuts[1], P_Lmin=self.L_cuts[2], P_Lmax=self.L_cuts[3], iter=iter, recalc_N0=True)
        return Cl/(Cl + N0)

    def get_response(self, typ, curl=False):
        qe_key = self._get_qe_key(typ, curl=curl)
        return self.qresp_lib.get_response(qe_key, 'p')

    def get_RDN0(self, typ, curl=False):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=curl)
        N0_unnorm = self.rdn0_lib.get_sim_nhl(-1, qe_key, qe_key)
        resp = self.get_response(typ, curl)
        qnorm = utils.cli(resp)**2
        return N0_unnorm * qnorm[:np.size(N0_unnorm)]

    def _get_Cl_phi(self):
        ells = np.arange(self.Lmax_map + 1)
        Cl_kappa = self.power.get_kappa_ps(ells, use_weyl=False)
        Cl_phi = 4/(ells*(ells+1))**2 * Cl_kappa
        Cl_phi[0] = 0
        return Cl_phi
    
    def _get_Cl_curl(self, use_cache=True):
        if use_cache:
            ells = np.load(f"../cache/_C_omega/Ls.npy")[1:]
            Cl_omega = np.load(f"../cache/_C_omega/C_omega.npy")[1:]
        else:
            ells = np.arange(1, self.Lmax_map + 1)
            Cl_omega = postborn.omega_ps(ells)
        Cl_curl = np.zeros(np.size(ells) + 1)
        Cl_curl[1:] = 4/(ells * (ells + 1))**2 * Cl_omega
        return Cl_curl

    def get_N1(self, typ):
        qe_key = self._get_qe_key(typ)
        fal = self._get_filt_matrix(self.cl_len, True) if self.gmv else fal = self.filt_dict
        lib_n1 = n1_fft.n1_fft(fal, self.cl_grad, self.cl_grad, self._get_Cl_phi(), lminbox=10, lmaxbox=5000 + 100) 
        Ls_n1 = np.linspace(30, 3000, 200)
        n1 = np.array([lib_n1.get_n1(qe_key, L, do_n1mat=False) for L in Ls_n1])
        qnorm = utils.cli(self.get_response(typ, curl=False))
        qnorm = InterpolatedUnivariateSpline(np.arange(np.size(qnorm)), qnorm)(Ls_n1)
        return Ls_n1, n1 * qnorm**2

    def get_map(self, typ, with_noise=True):
        if typ.lower() == "T":
            Tmap = self.maps_lib.get_sim_tmap(-1)
            return Tmap if with_noise else Tmap - self.get_noise_map("T")
        if typ.lower()== "E":
            Emap = self.maps_lib.get_sim_pmap(-1)[0]
            return Emap if with_noise else Emap - self.get_noise_map("E")
        if typ.lower() == "B":
            Bmap = self.maps_lib.get_sim_pmap(-1)[1]
            return Bmap if with_noise else Bmap - self.get_noise_map("B")
        raise ValueError("Type not of accepted T, E, or B.")
    
    def get_noise_map(self, typ):
        if typ.lower() == "t":
            return self.maps_lib.get_sim_tnoise(-1)
        if typ.lower() == "e":
            return self.maps_lib.get_sim_enoise(-1)
        if typ.lower() == "b":
            return self.maps_lib.get_sim_bnoise(-1)
        raise ValueError("Type not of accepted T, E, or B.")

