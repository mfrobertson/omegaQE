import numpy as np
import omegaqe
import DEMNUnii
from omegaqe.fisher import Fisher
from DEMNUnii.demnunii import Demnunii
from DEMNUnii.reconstruction import Reconstruction
from DEMNUnii.template import Template
from datetime import datetime
import copy


class Fields:

    def __init__(self, exp, use_lss_cache=False, use_cmb_cache=False, cmb_sim=0, deflect_typ="dem_dem", nthreads=1, gauss_lss=False, len_lss=False):
        self.nthreads = nthreads
        self.dm = Demnunii(nthreads)
        self.exp = exp
        self.fish = Fisher(exp, "TEB", True, "gradient", (30, 3000, 30, 5000), False, False, data_dir=omegaqe.DATA_DIR, cosmology=self.dm.cosmo)
        self.sim = cmb_sim
        self.deflect_typ = deflect_typ
        self.nside = self.dm.nside
        self.Lmax_map = self.dm.Lmax_map
        self.fields = "kgI"
        self.ells = np.arange(self.Lmax_map+1)
        self._initialise(use_lss_cache, use_cmb_cache, gauss_lss, len_lss)

    def _initialise(self, use_lss_cache, use_cmb_cache, gauss_lss, len_lss):
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True, use_cache=use_lss_cache, lensed=len_lss)
            self.fft_noise_maps[field] = self.get_noise_map(field, set_seed=True, fft=True)
        if gauss_lss:
            print("Using Cls of previous maps to generate new gaussian realisations.")
            self.y = self._get_y(input_kappa_map=self.dm.sht.read_map(f"{DEMNUnii.SIMS_DIR}/kappa_diff.fits"))
            for field in self.fields:
                self.fft_maps[field] = self.get_map(field, fft=True, use_cache=use_lss_cache, gaussian=gauss_lss)
        if use_cmb_cache:
            self.rec = None
        else:
            self.setup_rec(self.sim, self.deflect_typ)

    def get_Cl(self, typ, smoothing_nbins=None):
        alm1 = self.fft_maps[typ[0]]
        alm2 = self.fft_maps[typ[1]]
        return self.dm.sht.alm2cl(alm1, alm2, smoothing_nbins=smoothing_nbins)

    def _get_cov(self, N_fields):
        C = np.empty((self.Lmax_map, N_fields, N_fields))
        for iii, field_i in enumerate(self.fields):
            for jjj, field_j in enumerate(self.fields):
                C[:, iii, jjj] = self.get_Cl(field_i + field_j, smoothing_nbins=150)[1:]
        return C

    def _get_gauss_alm(self):
        return self.dm.sht.synalm(np.ones(self.Lmax_map + 1), self.Lmax_map)

    def _get_gauss_alms(self, Nfields):
        alms = np.empty((self.dm.sht.get_alm_size(), Nfields, 1), dtype="complex128")
        for iii in np.arange(Nfields):
            alms[:, iii, 0] = self._get_gauss_alm()
        return alms

    def _get_L(self, Cov, N_fields):
        L = np.linalg.cholesky(Cov)
        new_L = np.empty((self.Lmax_map + 1, N_fields, N_fields))
        new_L[1:, :, :] = L
        new_L[0, :, :] = 0.0
        return new_L

    def _matmul(self, L, v):
        rows = np.shape(L)[1]
        cols = np.shape(v)[2]
        res = np.empty((np.shape(v)[0], rows, cols), dtype="complex128")
        for row in np.arange(rows):
            for col in np.arange(cols):
                for iii in np.arange(np.shape(v)[1]):
                    res[:, row, col] += self.dm.sht.almxfl(v[:, iii, col], L[:, row, iii])
        return res

    def _get_y(self, input_kappa_map):
        N_fields = np.size(list(self.fields))
        C = self._get_cov(N_fields)
        L = self._get_L(C, N_fields)
        v = self._get_gauss_alms(N_fields)
        if input_kappa_map is not None:
            # TODO: this only works if k is the fist field listed
            C_kappa_sqrt = L[:, 0, 0]
            C_kappa_sqrt_inv = np.zeros(np.size(C_kappa_sqrt))
            C_kappa_sqrt_inv[1:] = 1/C_kappa_sqrt[1:]
            v[:, 0, 0] = self.dm.sht.almxfl(self.dm.sht.map2alm(input_kappa_map), C_kappa_sqrt_inv)
        y = self._matmul(L, v)
        return y

    def setup_rec(self, sim, deflect_typ, iter=False):
        self.sim = sim
        self.deflect_typ = deflect_typ
        print(f"Creating Reconstruction instance with exp: {self.exp}, and file: {self.dm.sims_dir}/{self.deflect_typ}/TQU_{self.sim}.fits")
        self.rec = Reconstruction(self.exp, filename=f"{self.dm.sims_dir}/{self.deflect_typ}/TQU_{self.sim}.fits", sim=self.sim, nthreads=self.nthreads, iter=iter)

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, iter_ext=None, data_dir=None):
        return self.fish.setup_noise(exp, qe, gmv, ps, L_cuts, iter, iter_ext, data_dir)

    def get_cached_cmb_lens_rec(self, typ, cmb_fields, sim=None, deflect_typ=None, iter=False):
        sim = self.sim if sim is None else sim
        deflect_typ = self.deflect_typ if deflect_typ is None else deflect_typ
        qe_typ = cmb_fields + "_iter" if iter else cmb_fields
        print(f"Getting cached {typ} reconstruction for sim: {sim}, deflection type: {deflect_typ}, and exp: {self.exp}.")
        return self.dm.sht.read_map(f"{self.dm.sims_dir}/{deflect_typ}/{self.exp}/{typ}/{qe_typ}_{sim}.fits")

    def get_cached_lss(self, field, gaussian, lensed=False):
        if gaussian:
            print(f"  Using cached Gaussian {field} map.")
            return self.dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}_maps/{field}_gaussian.fits")
        if lensed and field != "k":
            print(f"  Using cached lensed {field} map.")
            return self.dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}_maps/{field}_len.fits")
        return self.dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}_maps/{field}.fits")

    def _lensing_fac(self):
        return -self.ells*(self.ells + 1)/2
    
    def _get_cmb_lens_rec_iter(self, typ, cmb_fields):
        itmax = 7
        if self.rec.iter_rec_data is None:
            self.rec.calc_iter_rec(cmb_fields, itmax)
        rec_func = self.rec.get_phi_rec_iter if typ == "kappa" else self.rec.get_curl_rec_iter
        return rec_func(itmax)
    
    def _get_cmb_lens_rec(self, typ, cmb_fields, iter, fft):
        if self.rec is None:
            map = self.get_cached_cmb_lens_rec(typ, cmb_fields, iter=iter)
            if fft:
                return self.dm.sht.map2alm(map, nthreads=self.nthreads)
            return map
        if iter:
            potential_alm = self._get_cmb_lens_rec_iter(typ, cmb_fields)
        else:
            rec_func = self.rec.get_phi_rec if typ == "kappa" else self.rec.get_curl_rec
            potential_alm = rec_func(cmb_fields)
        alm = self.dm.sht.almxfl(potential_alm, self._lensing_fac())
        if fft:
            return alm
        return self.dm.sht.alm2map(alm, nthreads=self.nthreads)

    def get_kappa_rec(self, cmb_fields="T", iter=False, fft=False):
        return self._get_cmb_lens_rec("kappa", cmb_fields, iter, fft)

    def get_omega_rec(self, cmb_fields="T", iter=False, fft=False):
        return self._get_cmb_lens_rec("omega", cmb_fields, iter, fft)

    def _get_index(self, field):
        return np.where(np.char.array(list(self.fields)) == field)[0][0]

    def get_gaussian_map(self, field):
        index = self._get_index(field)
        alm = copy.deepcopy(self.y[:, index, 0])
        return self.dm.sht.alm2map(alm)

    def get_map(self, field, fft=False, use_cache=False, gaussian=False, lensed=False):
        if gaussian:
            print(f"Replacing {field} map with new gaussian realisation.")
            map = self.get_gaussian_map(field)
        elif use_cache:
            print(f"Using cache for map: {field}")
            map = self.get_cached_lss(field, gaussian, lensed=lensed)
        elif field == "k":
            map = self.dm.get_kappa_map()
        elif field == "g":
            map = self.dm.get_obs_gal_map(verbose=True, lensed=lensed)
        elif field == "I":
            map = self.dm.get_obs_cib_map(verbose=True, lensed=lensed)
        else:
            raise ValueError(f"Field typ {field} not expected.")
        if fft:
            return self.dm.sht.map2alm(map, nthreads=self.nthreads)
        return map

    def _get_N(self, field):
        kmax = 5000
        if field == "k":
            return self.fish.covariance.noise.get_N0("kappa", ellmax=kmax, recalc_N0=False)
        if field == "g":
            return self.fish.covariance.noise.get_gal_shot_N(ellmax=kmax)
        if field == "I":
            N_dust = self.fish.covariance.noise.get_dust_N(353e9, ellmax=kmax)
            N_cib = self.fish.covariance.noise.get_cib_shot_N(353e9, ellmax=kmax)
            N = N_dust+N_cib
            N[0] = 0
            return N
        nT, beam = self.fish.covariance.noise.get_noise_args(self.exp)
        return self.fish.covariance.noise.get_cmb_gaussian_N(field, nT, beam, kmax, exp=self.exp)

    def _get_seed(self, field=None, sim=None):
        if field is None:
            return int(datetime.now().timestamp())
        sim = self.sim if sim is None else sim
        seed = 3 * sim
        if field == "g":
            seed += 1
        elif field == "I":
            seed += 2
        return seed

    def get_noise_map(self, field, set_seed=False, fft=False):
        N = self._get_N(field)
        if set_seed:
            np.random.seed(self._get_seed(field))
        map = self.dm.sht.synfast(N)
        if fft:
            return self.dm.sht.map2alm(map, nthreads=self.nthreads)
        return map

    def omega_template(self, Nchi, Lmin=30, Lmax=3000, tracer_noise=False, use_kappa_rec=False, kappa_rec_qe_typ="TEB", neg_tracers=False):
        self.tem = Template(self, Lmin, Lmax, tracer_noise, use_kappa_rec, kappa_rec_qe_typ, neg_tracers=neg_tracers)
        return self.tem.get_omega(Nchi)
