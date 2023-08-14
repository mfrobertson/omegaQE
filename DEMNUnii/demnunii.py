from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
import numpy as np
from configparser import ConfigParser
import pandas as pd
import datetime
import re
from DEMNUnii.spherical import Spherical
import DEMNUnii

class Demnunii:

    def __init__(self, nthreads=1):
        self.data_dir = DEMNUnii.DATA_DIR
        self.cache_dir = DEMNUnii.CACHE_DIR
        self.sims_dir = DEMNUnii.SIMS_DIR
        self.config = self.setup_config()
        self.nside = int(self.parse_config(self.get_config("HealpixNside")))
        self.Lmax_map = DEMNUnii.LMAX_MAP
        self.sht = Spherical(self.nside, self.Lmax_map, nthreads=nthreads)
        self.snap_df = self.get_snap_info()
        self.cosmo = Cosmology("DEMNUnii")

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

    def parse_config(self, config):
        config = re.sub(r'(#.*)?\n?', '', str(config))
        return re.sub(r'(%.*)?\n?', '', config)

    def get_particle_mass(self):
        return float(self.get_config("UnitMass_in_g"))

    def get_snap(self, snap):
        sep = getFileSep()
        snap_num_app = "0" if len(str(snap)) == 2 else "00"
        filename = f'{self.data_dir}{sep}SurfaceMassDensities{sep}SurfDensMap_snap_{snap_num_app}{snap}.0.fits'
        return self.sht.read_map(filename)

    def get_density_snap(self, snap):
        pixel_area = self.sht.nside2pixarea()
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

    def get_density_map(self, zmin=0, zmax=1100, chi_min=None, chi_max=None, use_chi=False, verbose=False):
        if verbose: print(f"DEMNUnii: Constructing density map for zmin={zmin}, zmax={zmax}, chi_min={chi_min}, chi_max={chi_max}, use_chi={use_chi}")
        pixel_area = self.sht.nside2pixarea()
        npix = self.sht.nside2npix()
        particle_mass = self.get_particle_mass()
        rho = np.zeros(npix)
        snaps = self.get_snaps_chi(chi_min, chi_max) if use_chi else self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            rho += self.get_snap(snap) * particle_mass / pixel_area
            if verbose: print('\r', end='')
        if verbose: print("")
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

    def _window_LSST(self, z):
        return self.cosmo.gal_window_z(z)

    def _window_Planck(self, z):
        return self.cosmo.cib_window_z(z)

    def _window(self, snap, typ):
        zmin = self._get_zmin(snap)
        zmax = self._get_zmax(snap)
        zs = np.linspace(zmin, zmax, 1000)
        dz = zs[1] - zs[0]
        if typ == "LSST":
            return np.sum(self._window_LSST(zs)) * dz
        if typ == "Planck":
            return np.sum(self._window_Planck(zs)) * dz

    def get_obs_gal_map(self, zmin=0, zmax=1100, window="LSST", chi_min=None, chi_max=None, use_chi=False, verbose=False):
        if verbose: print(f"DEMNUnii: Constructing gal map for zmin={zmin}, zmax={zmax}, window={window}, chi_min={chi_min}, chi_max={chi_max}, use_chi={use_chi}")
        npix = self.sht.nside2npix()
        gal = np.zeros(npix)
        snaps = self.get_snaps_chi(chi_min, chi_max) if use_chi else self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            gal += self._window(snap, window) * self.get_density_snap(snap)
            if verbose: print('\r', end='')
        if verbose: print("")
        return gal

    def get_obs_cib_map(self, zmin=0, zmax=1100, window="Planck", chi_min=None, chi_max=None, use_chi=False, verbose=False):
        if verbose: print(f"DEMNUnii: Constructing CIB map for zmin={zmin}, zmax={zmax}, window={window}, chi_min={chi_min}, chi_max={chi_max}, use_chi={use_chi}")
        npix = self.sht.nside2npix()
        cib = np.zeros(npix)
        snaps = self.get_snaps_chi(chi_min, chi_max) if use_chi else self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            cib += self._window(snap, window) * self.get_density_snap(snap)
            if verbose: print('\r', end='')
        if verbose: print("")
        return cib

    def get_density_distro(self):
        snaps = self.get_snaps_z(0, 1100)
        distro = np.zeros(np.size(snaps))
        for iii, snap in enumerate(snaps):
            distro[iii] = np.sum(self.get_snap(snap))
        return distro


    def get_kappa_map(self, pb=False):
        sep = getFileSep()
        kappa_file = "map0_kappa_ecp262_dmn2_lmax8000.fits" if pb else "map0_kappa_ecp262_dmn2_lmax8000_first.fits"
        filename = self.data_dir+sep+"nbody"+sep+kappa_file
        return self.sht.read_map(filename)

    def get_omega_map(self):
        sep = getFileSep()
        omega_file = "map0_rotation_ecp262_dmn2_lmax8000.fits"
        filename = self.data_dir + sep + "nbody" + sep + omega_file
        return self.sht.read_map(filename)
