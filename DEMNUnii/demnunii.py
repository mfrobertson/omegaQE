from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
import numpy as np
from astropy.io import fits
import healpy as hp
from configparser import ConfigParser
import pandas as pd


DATA_DIR = "/mnt/lustre/users/astro/mr671/DEMNUnii/LCDM/"


class Demnunii:

    def __init__(self):
        self.data_dir = DATA_DIR
        self.config = self.setup_config()
        self.nside = 4096
        self.snap_df = self.get_snap_info()
        self.cosmo = Cosmology("DEMNUnii_params.ini")

    def setup_config(self):
        sep = getFileSep()
        paramfile = "paramfile.Surf.final.ini"
        with open(f'{self.data_dir}{sep}{paramfile}', 'r') as f:
            config_string = '[header]\n' + f.read()
        config = ConfigParser(delimiters=" ")
        config.read_string(config_string)
        return config

    def get_snap_info(self):
        sep = getFileSep()
        snap_info_file = "cosmoint_results_from_lcdm.txt"
        return pd.read_csv(self.data_dir+sep+snap_info_file, delim_whitespace=True)

    def get_config(self, name):
        return self.config.get("header", name)

    def get_particle_mass(self):
        return float(self.get_config("UnitMass_in_g"))

    def get_snap(self, snap):
        sep = getFileSep()
        snap_num_app = "0" if len(str(snap)) == 2 else "00"
        hdul = fits.open(f'{self.data_dir}{sep}SurfaceMassDensities{sep}SurfDensMap_snap_{snap_num_app}{snap}.0.fits')
        return np.asarray(hdul[1].data.base, dtype=float)

    def get_density_snap(self, snap):
        pixel_area = hp.pixelfunc.nside2pixarea(self.nside)
        particle_mass = self.get_particle_mass()
        rho = self.get_snap(snap) * particle_mass / pixel_area
        rho_bar = np.mean(rho)
        return rho / rho_bar - 1

    def get_snaps_z(self, zmin, zmax):
        zs = np.asarray(self.snap_df["6:zstart"])
        snaps = np.asarray(self.snap_df["#1:output"])
        return snaps[np.where(np.logical_and(zs >= zmin, zs <= zmax))]

    def get_snaps_chi(self, chi_min, chi_max):
        chis = np.asarray(self.snap_df["4:xstart"])
        snaps = np.asarray(self.snap_df["#1:output"])
        return snaps[np.where(np.logical_and(chis >= chi_min, chis <= chi_max))]

    def get_density_map(self, zmin=0, zmax=1100, chi_min=None, chi_max=None, use_chi=False):
        pixel_area = hp.pixelfunc.nside2pixarea(self.nside)
        npix = hp.nside2npix(self.nside)
        particle_mass = self.get_particle_mass()
        rho = np.zeros(npix)
        snaps = self.get_snaps_chi(chi_min, chi_max) if use_chi else self.get_snaps_z(zmin, zmax)
        for snap in snaps:
            rho += self.get_snap(snap) * particle_mass / pixel_area
        rho_bar = np.mean(rho)
        return rho / rho_bar - 1

    def _get_z(self, snap):
        return np.asarray(self.snap_df["2:z"][self.snap_df["#1:output"] == snap])[0]

    def _get_zmin(self, snap):
        return np.asarray(self.snap_df["6:zstart"][self.snap_df["#1:output"] == snap])[0]

    def _get_zmax(self, snap):
        return np.asarray(self.snap_df["7:z_end"][self.snap_df["#1:output"] == snap])[0]

    def _get_dChi(self, snap):
        return np.asarray(self.snap_df["8:dx"][self.snap_df["#1:output"] == snap])[0]

    def _bias(self, z, typ="unity"):
        if typ == "unity":
            return 1
        if typ == "LSST":
            return 1 + (0.84 * z)

    def _window_LSST(self, z):
        z0 = 0.311
        return 1 / (2 * z0) * (z / z0) ** 2 * np.exp(-z / z0) * self.cosmo.get_hubble(z) * self._bias(z, "LSST")

    def _window(self, snap, typ):
        zmin = self._get_zmin(snap)
        zmax = self._get_zmax(snap)
        if typ == "LSST":
            zs = np.linspace(zmin, zmax, 1000)
            dz = zs[1] - zs[0]
            return np.sum(self._window_LSST(zs)) * dz

    def _get_Ng(self, zmin=0, zmax=1100, window="LSST"):
        zs = np.linspace(zmin, zmax, 1000)
        dz = zs[1] - zs[0]
        if window == "LSST":
            return np.sum(self._window_LSST(zs)) * dz

    def get_obs_gal_map(self, zmin=0, zmax=1100, window="LSST", chi_min=None, chi_max=None, use_chi=False):
        pixel_area = hp.pixelfunc.nside2pixarea(self.nside)
        npix = hp.nside2npix(self.nside)
        particle_mass = self.get_particle_mass()
        gal = np.zeros(npix)
        snaps = self.get_snaps_chi(chi_min, chi_max) if use_chi else self.get_snaps_z(zmin, zmax)
        for snap in snaps:
            snap_map = self.get_snap(snap)
            gal += self._get_dChi(snap) * self._window(snap, window) * snap_map/np.sum(snap_map) * particle_mass / pixel_area
        gal_bar = np.mean(gal)
        return gal / gal_bar - 1

    def get_density_distro(self):
        snaps = self.get_snaps_z(0, 1100)
        distro = np.zeros(np.size(snaps))
        for iii, snap in enumerate(snaps):
            distro[iii] = np.sum(self.get_snap(snap))
        return distro

    def get_ps(self, map, lmax=4000):
        return hp.sphtfunc.anafast(map, lmax=lmax, use_pixel_weights=True)

    def _get_map_from_split_fits(self, filename):
        hdul = fits.open(filename)
        data = np.asarray(hdul[1].data.base)
        npix = hp.nside2npix(self.nside)
        map = np.zeros(npix)
        for iii in np.arange(np.size(data)):
            map[iii * 1024:(iii + 1) * 1024] = data[iii][0]
        return map

    def get_kappa_map(self, pb=False):
        sep = getFileSep()
        kappa_file = "map0_kappa_ecp262_dmn2_lmax8000.fits" if pb else "map0_kappa_ecp262_dmn2_lmax8000_first.fits"
        return self._get_map_from_split_fits(self.data_dir+sep+"nbody"+sep+kappa_file)

    def get_omega_map(self):
        sep = getFileSep()
        omega_file = "map0_rotation_ecp262_dmn2_lmax8000.fits"
        return self._get_map_from_split_fits(self.data_dir + sep + "nbody" + sep + omega_file)





