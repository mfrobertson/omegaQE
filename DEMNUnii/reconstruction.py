import os
import healpy as hp
import numpy as np
from plancklens.filt import filt_simple
from plancklens import utils
from plancklens import qest, qresp, nhl
from plancklens.n1 import n1
from plancklens.sims import phas, maps
from DEMNUnii.demnunii import Demnunii
from omegaqe.noise import Noise
from omegaqe.powerspectra import Powerspectra
import shutil


class Reconstruction:

    class MySimLib:

        def __init__(self, plancklens_mapslib, nside):
            self.plancklens_mapslib = plancklens_mapslib
            self.nside = nside
            self._shuffle = {idx: -1 for idx in range(-1, 300)}

        def get_sim_tmap(self, idx):
            return hp.sphtfunc.alm2map(self.plancklens_mapslib.get_sim_tmap(int(self._shuffle[-1])), self.nside)

        def get_sim_pmap(self, idx):
            elm, blm = self.plancklens_mapslib.get_sim_pmap(int(self._shuffle[-1]))
            return hp.alm2map_spin([elm, blm], self.nside, 2, hp.Alm.getlmax(elm.size))

        def hashdict(self):
            return {'sim_lib': self.plancklens_mapslib.hashdict(), 'shuffle': self.plancklens_mapslib.hashdict()}

    class MyMapLib:

        def __init__(self, lmax_map, maps_filename):
            self.lmax = lmax_map
            self.maps_filename = maps_filename
            self.maps = hp.fitsfunc.read_map(f"{self.maps_filename}", (0, 1, 2))

        def get_sim_tlm(self, idx):
            return hp.sphtfunc.map2alm(self.maps[0], lmax=self.lmax, mmax=self.lmax, use_pixel_weights=True)

        def _get_map_eblm(self, idx):
            return hp.sphtfunc.map2alm_spin(self.maps[1:], 2, lmax=self.lmax, mmax=self.lmax)

        def get_sim_elm(self, idx):
            return self._get_map_eblm(idx)[0]

        def get_sim_blm(self, idx):
            return self._get_map_eblm(idx)[1]

        def hashdict(self):
            return {'cmbs': self.maps_filename, 'noise': self.maps_filename, 'data': self.maps_filename}

    def __init__(self, exp, filename=None, L_cuts=(30, 3000, 30, 5000), sim=None):
        self.filename = filename
        self.setup_env()
        self.temp = os.path.join(os.environ['PLENS'], 'demnunii', str(sim))
        self.exp = exp
        self.dm = Demnunii()
        self.noise = Noise()
        self.L_cuts = L_cuts
        self.Lmax_map = self.dm.Lmax_map
        self.indices = ["tt", "ee", "bb"]
        self.cl_len, self.cl_grad = self.get_cmb_cls()
        self.noise_cls = self.get_noise_cls(exp)
        self._initialise()
        self.setup = False
        if filename is not None:
            self.setup_reconstruction(self.filename)

    def _initialise(self):
        print("Initialising filters and librarys for Plancklens")
        self.transfer_dict = {idx[0]: np.ones(self.Lmax_map + 1) for idx in self.indices}
        filters = self.get_filters(self.indices, self.noise_cls)
        self.filt_dict = {'t': filters[0], 'e': filters[1], 'b': filters[2]}
        self.qresp_lib = qresp.resp_lib_simple(os.path.join(self.temp, 'qresp'), self.Lmax_map, self.cl_grad, self.cl_grad, self.filt_dict, self.Lmax_map)
        self.n1_lib = n1.library_n1(os.path.join(self.temp, 'n1'), self.cl_grad['tt'], self.cl_grad['ee'], self.cl_grad['bb'], lmaxphi=5000)

    def get_Lmax_filt(self):
        return np.max(self.L_cuts)

    def _get_Lmin(self, typ):
        if typ == "tt":
            return self.L_cuts[0]
        if typ == "ee" or typ == "bb":
            return self.L_cuts[2]

    def _get_Lmax(self, typ):
        if typ == "tt":
            return self.L_cuts[1]
        if typ == "ee" or typ == "bb":
            return self.L_cuts[3]

    def setup_env(self):
        if not 'PLENS' in os.environ.keys():
            plancklens_cachedir = "_tmp"
            os.environ['PLENS'] = plancklens_cachedir
            print(f"Setting up Plancklens ache at {plancklens_cachedir}")
        if os.path.exists(os.environ['PLENS']) and os.path.isdir(os.environ['PLENS']):
            print(f"Removing existing plancklens cache at {os.environ['PLENS']}")
            shutil.rmtree(os.environ['PLENS'])


    def get_cmb_cls(self):
        Tcmb = 2.7255
        fac = (Tcmb * 1e6) ** 2
        indices = self.indices + ['te']
        cl_len = {idx : self.dm.cosmo.get_lens_ps(idx.upper()) * fac for idx in indices}
        cl_grad = {idx: self.dm.cosmo.get_grad_lens_ps(idx.upper()) * fac for idx in indices}
        return cl_len, cl_grad

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

    def get_filters(self, indices, noise_cls=None):
        Lmax_filt = self.get_Lmax_filt()
        if Lmax_filt >= self.Lmax_map:
            Lmax_filt = self.Lmax_map - 1
        filts = [utils.cli(self.cl_len[idx][:Lmax_filt + 1] + noise_cls[idx[0]][:Lmax_filt + 1]) if noise_cls is not None else
                utils.cli(self.cl_len[idx][:Lmax_filt + 1]) for idx in indices]
        return self._apply_cuts(filts, indices)

    def _raise_typ_error(self, typ):
        raise ValueError(f"QE type {typ} not recognized. Recognized types included TEB (for mv), EB (for pol only), or TT.")

    def _get_qe_key(self, typ, curl=False):
        potential = "x" if curl else "p"
        if typ == "TEB":
            return potential
        if typ == "T":
            return potential + "tt"
        if typ == "EB":
            return potential + "_p"
        self._raise_typ_error(typ)

    def setup_reconstruction(self, TQUmaps_filename):
        print(f"Setting up reconstruction for file: {TQUmaps_filename}")
        libdir_pixphas = os.path.join(self.temp, 'phas_lmax%s' % self.Lmax_map)
        pix_phas = phas.lib_phas(libdir_pixphas, 3, self.Lmax_map)
        maps_lib = maps.cmb_maps_harmonicspace(self.MyMapLib(self.Lmax_map, TQUmaps_filename), self.transfer_dict, self.noise_cls, noise_phas=pix_phas)
        self.sim_lib = self.MySimLib(maps_lib, self.dm.nside)
        weighted_maps_lib = filt_simple.library_fullsky_sepTP(os.path.join(self.temp, 'ivfs'), self.sim_lib, self.dm.nside, self.transfer_dict, self.cl_grad, self.filt_dict['t'], self.filt_dict['e'], self.filt_dict['b'])
        self.qlms_lib = qest.library_sepTP(os.path.join(self.temp, 'qlms_dd'), weighted_maps_lib, weighted_maps_lib, self.cl_grad['te'], self.dm.nside, lmax_qlm=self.Lmax_map)
        self.rdn0_lib = nhl.nhl_lib_simple(os.path.join(self.temp, 'rdn0'), weighted_maps_lib, self.cl_grad, self.Lmax_map)
        self.setup = True

    def _check_setup(self):
        if not self.setup:
            raise ValueError("Need to call 'setup_reconstruction' first. No qlm_lib has been instantiated.")

    def get_phi_rec(self, typ):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=False)
        qlm = self.qlms_lib.get_sim_qlm(qe_key, -1)
        resp = self.qresp_lib.get_response(qe_key, 'p')
        qnorm = utils.cli(resp)
        return hp.almxfl(qlm, qnorm)

    def get_curl_rec(self, typ):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=True)
        qlm = self.qlms_lib.get_sim_qlm(qe_key, -1)
        resp = self.qresp_lib.get_response(qe_key, 'p')
        qnorm = utils.cli(resp)
        return hp.almxfl(qlm, qnorm)

    def get_response(self, typ, curl=False):
        qe_key = self._get_qe_key(typ, curl=curl)
        return self.qresp_lib.get_response(qe_key, 'p')

    def get_RDN0(self, typ, curl=False):
        self._check_setup()
        qe_key = self._get_qe_key(typ, curl=curl)
        N0_unnorm = self.rdn0_lib.get_sim_nhl(-1, qe_key, qe_key)
        resp = self.get_response(typ, curl)
        return N0_unnorm * resp[:np.size(N0_unnorm)]**2

    def _get_Cl_phi(self):
        power = Powerspectra()
        power.cosmo = self.dm.cosmo
        ells = np.arange(1, self.Lmax_map + 1)
        Cl_phi = np.zeros(np.size(ells) + 1)
        Cl_phi[1:] = power.get_phi_ps(ells)
        return Cl_phi

    def get_N1(self, typ, curl=False, lmax=3000):
        Cl_phi = self._get_Cl_phi()
        qe_key = self._get_qe_key(typ, curl=curl)
        N1_unnorm = self.n1_lib.get_n1(qe_key, "p", Cl_phi, self.filt_dict['t'], self.filt_dict['e'], self.filt_dict['b'], lmax)
        resp = self.get_response(typ, curl)
        return N1_unnorm * resp[:np.size(N1_unnorm)] ** 2
