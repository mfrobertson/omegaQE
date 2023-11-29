from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
from omegaqe.powerspectra import Powerspectra
import numpy as np
from fullsky_sims.spherical import Spherical
import fullsky_sims
import healpy as hp

class Agora:

    def __init__(self, nthreads=1):
        self.data_dir = "/mnt/lustre/users/astro/mr671/AGORA"
        self.cache_dir = f"{fullsky_sims.CACHE_DIR}/../cache_ag"
        self.sims_dir = f"{self.data_dir}/cmb_sims"
        self.reset_default_dirs()
        self.nside = 8192
        self.Lmax_map = fullsky_sims.LMAX_MAP
        self.sht = Spherical(self.nside, self.Lmax_map, nthreads=nthreads)
        self.cosmo = Cosmology("AGORA")
        self._setup_dndz_splines()
        self.power = Powerspectra(cosmology=self.cosmo)

    def _setup_dndz_splines(self):
        nbins = 5
        zs, dn_dz1 = self.get_gal_dndz(1)
        dn_dzs = np.empty((5, np.size(dn_dz1)))
        dn_dzs[0] = dn_dz1
        for bin in np.arange(1, nbins):
            zs, dn_dz = self.get_gal_dndz(bin+1)
            dn_dzs[bin] = dn_dz
        self.cosmo.setup_dndz_splines(zs, dn_dzs, biases=[1.23,1.36,1.5,1.65,1.8])

    def reset_default_dirs(self):
        fullsky_sims.DATA_DIR = self.data_dir
        fullsky_sims.CACHE_DIR = self.cache_dir
        fullsky_sims.SIMS_DIR = self.sims_dir

    def _lensing_fac(self):
        ells = np.arange(self.Lmax_map+1)
        return ells*(ells + 1)/2

    def get_kappa_map(self, pb=True):
        # phi_lm = hp.read_alm(f"{self.data_dir}/phi/agora_phiNG_phi1_seed1.alm")
        # kappa_lm = hp.almxfl(phi_lm, self._lensing_fac())
        # return hp.alm2map(kappa_lm, self.nside)
        if not pb:
            raise ValueError("Only have postborn kappa map...")
        kappa_map, _, _ = self.sht.read_map(f"{self.data_dir}/phi/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")
        return kappa_map
    
    def get_omega_map(self):
        raise ValueError("AGORA has no omega map.")

    def get_gal_map(self, bin=1):
        return self.sht.read_map(f"{self.data_dir}/gal/agora_biaseddensity_lsst_y1_lens_zbin{bin}_fullsky.fits")
    
    def get_cib_map(self, nu=353):
        return self.sht.read_map(f"{self.data_dir}/cib/agora_len_mag_cibmap_planck_{nu}ghz.fits")
    
    def get_gal_dndz(self, bin=1):
        zs = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=0, skiprows=2)
        dndz = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=bin, skiprows=2)
        return zs, dndz