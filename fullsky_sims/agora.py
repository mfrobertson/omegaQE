from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
from omegaqe.powerspectra import Powerspectra
import numpy as np
from fullsky_sims.spherical import Spherical
import healpy as hp
from scipy.constants import Planck, physical_constants

class Agora:

    def __init__(self, nthreads=1):
        self.data_dir = "/mnt/lustre/users/astro/mr671/AGORA/"
        self.cache_dir = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/cache_ag/"
        self.sims_dir = f"{self.data_dir}/cmb_sims/"
        self.omegaqe_data = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/data_ag/"
        self.nside = 8192
        self.Lmax_map = 5000
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

    def _lensing_fac(self):
        ells = np.arange(self.Lmax_map+1)
        return ells*(ells + 1)/2

    def get_kappa_map(self, pb=True, pixel_corr=True):
        if not pb:
            raise ValueError("Only have postborn kappa map...")
        kappa_map, _, _ = self.sht.read_map(f"{self.data_dir}/phi/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")
        if pixel_corr: kappa_map = self._apply_pixel_correction(kappa_map)
        return kappa_map
    
    def get_omega_map(self):
        raise ValueError("AGORA has no omega map.")

    def get_gal_bin_map(self, bin=1):
        return self.sht.read_map(f"{self.data_dir}/gal/agora_biaseddensity_lsst_y1_lens_zbin{bin}_fullsky.fits")
    
    def _apply_pixel_correction(self, map):
        alm = self.sht.map2alm(map)
        alm_corr = self.sht.almxfl(alm, 1/self.sht.pixwin)
        return self.sht.alm2map(alm_corr)
    
    def get_obs_gal_map(self, pixel_corr=True, lensed=True, verbose=False):
        if not lensed:
            raise ValueError("AGORA products are all lensed.")
        zs = np.linspace(0, 1200, 10000)
        dz = zs[1]-zs[0]
        z_distr_func = self.cosmo._get_z_distr_func("LSST_a")
        letters = list("abcde")
        gal_map = self.get_gal_bin_map(1)*np.sum(dz * z_distr_func(zs))
        for bin in np.arange(2,6):
            z_distr_func = self.cosmo._get_z_distr_func(f"LSST_{letters[bin-1]}")
            gal_map += self.get_gal_bin_map(bin)*np.sum(dz * z_distr_func(zs))
        z_distr_func = self.cosmo._get_z_distr_func("agora")
        gal_map /= np.sum(dz * z_distr_func(zs))
        if pixel_corr: gal_map = self._apply_pixel_correction(gal_map)
        return gal_map
    
    def get_obs_cib_map(self, nu=353, pixel_corr=True, lensed=True, verbose=False, Kelvin=True):
        if not lensed:
            raise ValueError("AGORA products are all lensed.")
        cib_map = self.sht.read_map(f"{self.data_dir}/cib/agora_len_mag_cibmap_planck_{nu}ghz.fits")
        if Kelvin: cib_map *= self.Jy_to_muK(nu*1e9)
        if pixel_corr: cib_map = self._apply_pixel_correction(cib_map)
        return cib_map
    
    def get_gal_dndz(self, bin=1):
        zs = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=0, skiprows=2)
        dndz = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=bin, skiprows=2)
        return zs, dndz
    
    def get_obs_ksz_map(self, pixel_corr=True, lensed=True, agn_T_pow=80):
        ksz = self.sht.read_map(f"{self.data_dir}/ksz/agora_lkszNG_bahamas{agn_T_pow}_bnd_unb_1.0e+12_1.0e+18_lensed.fits")
        if pixel_corr: ksz = self._apply_pixel_correction(ksz)
        return ksz
    
    @staticmethod
    def b_nu(nu):
        T = 2.7255
        h = Planck
        k_B = physical_constants["Boltzmann constant"][0]
        c = physical_constants["speed of light in vacuum"][0]
        fac = h*nu/(k_B*T)
        return (2 * h * nu**3) / (c**2 * (np.exp(fac) - 1)) * ((np.exp(fac))/(np.exp(fac) - 1)) * (fac/T)
    
    @staticmethod
    def gauss_pdf(x, mean, sig):
        # sig = np.sqrt(var)
        return 1/(sig * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mean)/sig)**2)
    
    def tau(self, nu_c, nu):
        return self.gauss_pdf(nu, nu_c, nu_c/10)

    
    def Jy_to_muK(self, nu, alpha=-1, use_tau=False):
        if use_tau:
            nus = np.linspace(0.1e9,1000e9,1000)
            dnu = nus[1] - nus[0]
            num = np.sum(self.b_nu(nus) * self.tau(nu, nus)) * dnu
            denom = np.sum(self.tau(nu, nus) * (nus/nu)**alpha) * dnu
            return (num/denom * 1e20)**-1
        return (self.b_nu(nu) * 1e20)**-1
    
    def y_to_muK(self, nu, use_tau=False):
        # For tau(nu) = delta_func in B4 2212.07420
        # TODO: use gauss for tau
        
        T = 2.7255
        h = Planck
        k_B = physical_constants["Boltzmann constant"][0]
        if use_tau:
            nus = np.linspace(0.1e9,2000e9,1000)
            dnu = nus[1] - nus[0]
            fac = h*nus/(k_B*T)
            num = np.sum(self.b_nu(nus) * self.tau(nu, nus)) * dnu
            denom = np.sum(self.b_nu(nus) * self.tau(nu, nus) * T * ( (fac * (np.exp(fac) + 1)) / (np.exp(fac) - 1) - 4 )) * dnu
            return (num/denom)**-1 * 1e6
        fac = h*nu/(k_B*T)
        return T * ( (fac * (np.exp(fac) + 1)) / (np.exp(fac) - 1) - 4 ) * 1e6

    
    def get_obs_y_map(self, pixel_corr=True, lensed=True, agn_T_pow=80):
        y_map = self.sht.read_map(f"{self.data_dir}/tsz/agora_ltszNG_bahamas{agn_T_pow}_bnd_unb_1.0e+12_1.0e+18_lensed.fits")
        if pixel_corr: y_map = self._apply_pixel_correction(y_map)
        return y_map
    
    def get_obs_tsz_map(self, nu, pixel_corr=True, lensed=True, agn_T_pow=80):
        return self.get_obs_y_map(pixel_corr, lensed, agn_T_pow) * self.y_to_muK(nu)
    
    def get_obs_rad_map(self, nu, pixel_corr=True, lensed=True):
        rad = self.sht.read_map(f"{self.data_dir}/radio/agora_radiomap_len_universemachine_trinity_{nu}ghz_randflux_datta2018_truncgauss.0.fits", field=0)
        rad[rad*self.sht.nside2pixarea()>6e-3]=0
        rad *= self.Jy_to_muK(nu*1e9, -0.75)
        if pixel_corr: rad = self._apply_pixel_correction(rad)
        return rad
    
    def get_PK(self):
        return self.cosmo.get_matter_PK(typ="matter")