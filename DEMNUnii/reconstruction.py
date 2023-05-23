import os
import healpy as hp
import numpy as np
from plancklens.filt import filt_simple
from plancklens import utils
from plancklens import qest, qresp
from plancklens.sims import phas, maps
from demnunii import Demnunii
from omegaqe.noise import Noise


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

    def __init__(self, exp, L_cuts=(30, 3000, 30, 5000)):
        self.setup_env()
        self.temp = os.path.join(os.environ['PLENS'], 'temp', 'idealized_example')
        self.exp = exp
        self.dm = Demnunii()
        self.noise = Noise()
        self.L_cuts = L_cuts
        self.lmax_map = 6000
        self.cl_len, self.cl_grad = self.get_cmb_cls()
        self.noise_cls = self.get_noise_cls(exp)

    def _get_Lmin(self, typ):
        if typ == "t":
            return self.L_cuts[0]
        if typ == "e" or typ == "b":
            return self.L_cuts[2]

    def _get_Lmax(self, typ):
        if typ == "t":
            return self.L_cuts[1]
        if typ == "e" or typ == "b":
            return self.L_cuts[3]

    def setup_env(self):
        if not 'PLENS' in os.environ.keys():
            os.environ['PLENS'] = '_tmp'

    def get_cmb_cls(self):
        Tcmb = 2.7255
        fac = (Tcmb * 1e6) ** 2
        indices = ['tt', 'ee', 'bb', 'te']
        cl_len = {idx : self.dm.cosmo.get_lens_ps(idx.upper()) * fac for idx in indices}
        cl_grad = {idx: self.dm.cosmo.get_grad_lens_ps(idx.upper()) * fac for idx in indices}
        return cl_len, cl_grad

    def get_noise_cls(self, exp):
        indices = ["tt", "ee", "bb"]
        Tcmb = 2.7255
        fac = (Tcmb * 1e6) ** 2
        return {idx[0]: self.noise.get_cmb_gaussian_N(idx.upper(), None, None, ellmax=self.lmax_map, exp=exp) * fac for idx in indices}

    def _apply_cuts(self, filts, indices):
        for iii, filt in enumerate(filts):
            filt[:self._get_Lmin(indices[iii])] *= 0.
            filt[self._get_Lmax(indices[iii]) + 1:] *= 0.
        return filts

    def get_filters(self, indices, noise_cls=None):
        filts = [(utils.cli(self.cl_len[idx][:self.lmax_map + 1] + noise_cls[idx[0]][:self.lmax_map + 1])) if noise_cls is not None else
                utils.cli(self.cl_len[idx][:self.lmax_map + 1]) for idx in indices]
        return self._apply_cuts(filts, indices)

    def _raise_typ_error(self, typ):
        raise ValueError(f"QE type {typ} not recognized. Recognized types included mv and t")

    def _get_indices(self, typ):
        if typ == "mv":
            return ['tt', 'ee', 'bb']
        if typ == "t":
            return ['tt']
        self._raise_typ_error(typ)

    def _get_qe_key(self, typ):
        if typ == "mv":
            return "p"
        if typ == "t":
            return "ptt"
        self._raise_typ_error(typ)

    def get_phi_rec(self, typ, TQUmaps_filename):
        indices = ['tt', 'ee', 'bb']
        filters = self.get_filters(indices, self.noise_cls)
        transfer_dict = {idx[0]:np.ones(self.lmax_map + 1) for idx in indices}
        libdir_pixphas = os.path.join(self.temp, 'phas_lmax%s' % self.lmax_map)
        pix_phas = phas.lib_phas(libdir_pixphas, 3, self.lmax_map)
        maps_lib = maps.cmb_maps_harmonicspace(self.MyMapLib(self.lmax_map, TQUmaps_filename), transfer_dict, self.noise_cls, noise_phas=pix_phas)
        self.sim_lib = self.MySimLib(maps_lib, self.dm.nside)
        sim_lib = self.sim_lib
        weighted_maps_lib = filt_simple.library_fullsky_sepTP(os.path.join(self.temp, 'ivfs'), sim_lib, self.dm.nside, transfer_dict, self.cl_len, filters[0], filters[1], filters[2])
        qlms_lib = qest.library_sepTP(os.path.join(self.temp, 'qlms_dd'), weighted_maps_lib, weighted_maps_lib, self.cl_len['te'], self.dm.nside, lmax_qlm=self.lmax_map)
        filt_dict = {'t': weighted_maps_lib.get_ftl(), 'e': weighted_maps_lib.get_fel(), 'b': weighted_maps_lib.get_fbl()}
        cl_weight = self.cl_len
        cl_weight['bb'] *= 0.
        qresp_lib = qresp.resp_lib_simple(os.path.join(self.temp, 'qresp'), self.lmax_map, cl_weight, self.cl_grad, filt_dict, self.lmax_map)
        qe_key = self._get_qe_key(typ)
        qlm = qlms_lib.get_sim_qlm(qe_key, -1)
        resp = qresp_lib.get_response(qe_key, 'p')
        qnorm = utils.cli(resp)
        return hp.almxfl(qlm, qnorm)
