import numpy as np
import omegaqe
from omegaqe.fisher import Fisher
from demnunii import Demnunii
from reconstruction import Reconstruction
import healpy as hp


class Fields:

    def __init__(self, exp, use_cache=False):
        self.fish = Fisher(exp, "TEB", True, "gradient", (30, 3000, 30, 5000), False, False, data_dir=omegaqe.DATA_DIR)
        self.exp = exp
        self.dm = Demnunii()
        self.rec = Reconstruction(self.exp, filename="/mnt/lustre/users/astro/mr671/len_cmbs/sims/demnunii/TQU_0.fits")
        self.nside = self.dm.nside
        self.Lmax_map = self.dm.Lmax_map
        self.fields = "kgI"
        self.ells = np.arange(self.Lmax_map)
        self._initialise(use_cache)

    def _initialise(self, use_cache):
        self.fft_maps = dict.fromkeys(self.fields)
        self.fft_noise_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True, use_cache=use_cache)
            self.fft_noise_maps[field] = self.get_noise_map(field, fft=True)

    def setup_noise(self, exp=None, qe=None, gmv=None, ps=None, L_cuts=None, iter=None, iter_ext=None, data_dir=None):
        return self.fish.setup_noise(exp, qe, gmv, ps, L_cuts, iter, iter_ext, data_dir)

    def get_kappa_rec(self, cmb_fields="T"):
        kappa_map = 2 * np.pi * self.rec.get_phi_rec(cmb_fields)
        return hp.almxfl(kappa_map, self.ells*(self.ells + 1)/2)

    def get_map(self, field, fft=False, use_cache=False):
        if use_cache:
            map = hp.fitsfunc.read_map(f"_maps/{field}.fits")
        elif field == "k":
            map = self.dm.get_kappa_map()
        elif field == "g":
            map = self.dm.get_obs_gal_map(verbose=True)
        elif field == "I":
            map = self.dm.get_obs_cib_map(verbose=True)
        else:
            raise ValueError(f"Field typ {field} not expected.")
        if fft:
            return hp.map2alm(map, lmax=self.Lmax_map, mmax=self.Lmax_map, use_pixel_weights=True)
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
        map = hp.sphtfunc.synfast(N, self.nside, self.Lmax_map, self.Lmax_map)
        if fft:
            return hp.map2alm(map, lmax=self.Lmax_map, mmax=self.Lmax_map, use_pixel_weights=True)
        return map


