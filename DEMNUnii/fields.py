import numpy as np
import omegaqe
import DEMNUnii
from omegaqe.fisher import Fisher
from DEMNUnii.demnunii import Demnunii
from DEMNUnii.reconstruction import Reconstruction
from DEMNUnii.template import Template


class Fields:

    def __init__(self, exp, use_lss_cache=False, use_cmb_cache=False, cmb_sim=0, deflect_typ="dem_dem", nthreads=1):
        self.nthreads = nthreads
        self.fish = Fisher(exp, "TEB", True, "gradient", (30, 3000, 30, 5000), False, False, data_dir=omegaqe.DATA_DIR)
        self.exp = exp
        self.dm = Demnunii(nthreads)
        self.sim = cmb_sim
        self.deflect_typ = deflect_typ
        self.nside = self.dm.nside
        self.Lmax_map = self.dm.Lmax_map
        self.fields = "kgI"
        self.ells = np.arange(self.Lmax_map+1)
        self._initialise(use_lss_cache, use_cmb_cache)

    def _initialise(self, use_lss_cache, use_cmb_cache):
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True, use_cache=use_lss_cache)
            self.fft_noise_maps[field] = self.get_noise_map(field, fft=True)
        if use_cmb_cache:
            self.rec = None
        else:
            self.setup_rec(self.sim, self.deflect_typ)

    def setup_rec(self, sim, deflect_typ):
        self.sim = sim
        self.deflect_typ = deflect_typ
        print(f"Creating Reconstruction instance with exp: {self.exp}, and file: {self.dm.sims_dir}/{self.deflect_typ}/TQU_{self.sim}.fits")
        self.rec = Reconstruction(self.exp, filename=f"{self.dm.sims_dir}/{self.deflect_typ}/TQU_{self.sim}.fits", sim=self.sim, nthreads=self.nthreads)

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, iter_ext=None, data_dir=None):
        return self.fish.setup_noise(exp, qe, gmv, ps, L_cuts, iter, iter_ext, data_dir)

    def get_cached_cmb_lens(self, typ, cmb_fields, sim=None):
        sim = self.sim if sim is None else sim
        return self.dm.sht.read_map(f"{self.dm.sims_dir}/{self.deflect_typ}/{self.exp}/{typ}/{cmb_fields}_{sim}.fits")

    def get_cached_lss(self, field):
        return self.dm.sht.read_map(f"{DEMNUnii.CACHE_DIR}_maps/{field}.fits")

    def _lensing_fac(self):
        return -self.ells*(self.ells + 1)/2

    def get_kappa_rec(self, cmb_fields="T", fft=False):
        if self.rec is None:
            kappa_map = self.get_cached_cmb_lens("kappa", cmb_fields)
            if fft:
                return self.dm.sht.map2alm(kappa_map, nthreads=self.nthreads)
            return kappa_map
        phi_alm = self.rec.get_phi_rec(cmb_fields)
        kappa_alm = self.dm.sht.almxfl(phi_alm, self._lensing_fac())
        if fft:
            return kappa_alm
        return self.dm.sht.alm2map(kappa_alm, nthreads=self.nthreads)

    def get_omega_rec(self, cmb_fields="T", fft=False):
        if self.rec is None:
            omega_map = self.get_cached_cmb_lens("omega", cmb_fields)
            if fft:
                return self.dm.sht.map2alm(omega_map, nthreads=self.nthreads)
            return omega_map
        curl_alm = self.rec.get_curl_rec(cmb_fields)
        omega_alm = self.dm.sht.almxfl(curl_alm, self._lensing_fac())
        if fft:
            return omega_alm
        return self.dm.sht.alm2map(omega_alm, nthreads=self.nthreads)

    def get_map(self, field, fft=False, use_cache=False):
        if use_cache:
            print(f"Using cache for map: {field}")
            map = self.get_cached_lss(field)
        elif field == "k":
            map = self.dm.get_kappa_map()
        elif field == "g":
            map = self.dm.get_obs_gal_map(verbose=True)
        elif field == "I":
            map = self.dm.get_obs_cib_map(verbose=True)
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

    def get_noise_map(self, field, set_seed=False, fft=False):
        N = self._get_N(field)
        if set_seed:
            np.random.seed(99)
        map = self.dm.sht.synfast(N)
        if fft:
            return self.dm.sht.map2alm(map, nthreads=self.nthreads)
        return map

    def omega_template(self, Nchi, Lmin=30, Lmax=3000, tracer_noise=False, use_kappa_rec=False, kappa_rec_qe_typ="TEB"):
        self.tem = Template(self, Lmin, Lmax, tracer_noise, use_kappa_rec, kappa_rec_qe_typ)
        return self.tem.get_omega(Nchi)


