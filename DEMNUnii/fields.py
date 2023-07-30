import numpy as np
import omegaqe
from omegaqe.fisher import Fisher
from demnunii import Demnunii
from reconstruction import Reconstruction
import healpy as hp


class Fields:

    def __init__(self, exp):
        self.fish = Fisher(exp, "TEB", True, "gradient", (30, 3000, 30, 5000), False, False, data_dir=omegaqe.DATA_DIR)
        self.exp = exp
        self.dm = Demnunii()
        self.rec = Reconstruction(self.exp, filename="/mnt/lustre/users/astro/mr671/len_cmbs/sims/demnunii/TQU_0.fits")
        self.N_pix = self.dm.nside
        self.Lmax_map = self.dm.Lmax_map
        self.fields = "kgI"
        self.ells = np.arange(self.Lmax_map)
        self.fft_maps = dict.fromkeys(self.fields)
        for field in self.fields:
            self.fft_maps[field] = self.get_map(field, fft=True)

    def get_kappa_rec(self, cmb_fields="T"):
        kappa_map = 2 * np.pi * self.rec.get_phi_rec(cmb_fields)
        return hp.almxfl(kappa_map, self.ells**2/2)

    def get_map(self, field, fft=False):
        if field == "k":
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

