from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
from omegaqe.powerspectra import Powerspectra
import numpy as np
from configparser import ConfigParser
import pandas as pd
import datetime
import re
from fullsky_sims.spherical import Spherical
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, UnivariateSpline
from scipy import signal

class Demnunii:

    def __init__(self, nthreads=1):
        self.data_dir = "/mnt/lustre/users/astro/mr671/DEMNUnii/LCDM/"
        self.cache_dir = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/cache/"
        self.sims_dir = f"/mnt/lustre/users/astro/mr671/len_cmbs/sims/"
        self.omegaqe_data = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/data/"
        self.config = self.setup_config()
        self.nside = int(self.parse_config(self.get_config("HealpixNside")))
        self.Lmax_map = 5000
        self.sht = Spherical(self.nside, self.Lmax_map, nthreads=nthreads)
        self.snap_df = self.get_snap_info()
        self.cosmo = Cosmology("DEMNUnii")
        self.power = Powerspectra(cosmology=self.cosmo)
        self.power.matter_PK = self.get_PK()

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

    def get_particle_snap(self, snap, lensed=False):
        sep = getFileSep()
        snap_num_app = "0" if len(str(snap)) == 2 else "00"
        if lensed:
            filename = f'{self.cache_dir}{sep}_len_snaps{sep}len_snap_{snap_num_app}{snap}.fits'
        else:
            filename = f'{self.data_dir}{sep}SurfaceMassDensities{sep}SurfDensMap_snap_{snap_num_app}{snap}.0.fits'
        return self.sht.read_map(filename)
    
    def get_kappa_snap_old(self, snap):
        sep = getFileSep()
        snap_num_app = "0" if len(str(snap)) == 2 else "00"
        filename = f'{self.data_dir}{sep}CMB_Convergence{sep}KappaMap_snap_{snap_num_app}{snap}.0.fits'
        return self.sht.read_map(filename)

    def get_density_snap(self, snap, lensed=False):
        rho = self.get_particle_snap(snap, lensed)
        rho_bar = np.mean(rho)
        return rho / rho_bar - 1

    def get_snaps_z(self, zmin, zmax):
        z_starts = np.asarray(self.snap_df["6:zstart"])
        z_ends = np.asarray(self.snap_df["7:z_end"])
        snaps = np.asarray(self.snap_df["#1:output"])
        indices = np.where(np.logical_and(z_starts >= zmin, z_ends <= zmax))[0]
        snaps = snaps[indices]
        print(f"Using particles between redshifts zmin: {z_starts[indices][0]} (snap {np.max(snaps)}) and zmax: {z_ends[indices][-1]} (snap {np.min(snaps)})")
        return snaps

    def get_density_map(self, zmin=0, zmax=1100, verbose=False):
        if verbose: print(f"DEMNUnii: Constructing density map for zmin={zmin}, zmax={zmax}")
        pixel_area = self.sht.nside2pixarea()
        npix = self.sht.nside2npix()
        particle_mass = self.get_particle_mass()
        rho = np.zeros(npix)
        snaps =  self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            rho += self.get_particle_snap(snap) * particle_mass / pixel_area
            if verbose: print('\r', end='')
        if verbose: print("")
        rho_bar = np.mean(rho)
        return rho / rho_bar - 1
    
    def _get_param_at_snap(self, param_name, snap):
        return np.asarray(self.snap_df[param_name][self.snap_df["#1:output"] == snap])[0]

    def _get_z(self, snap):
        return self._get_param_at_snap("2:z", snap)

    def _get_zmin(self, snap):
        return self._get_param_at_snap("6:zstart", snap)

    def _get_zmax(self, snap):
        return self._get_param_at_snap("7:z_end", snap)
    
    def _get_chi(self, snap):
        return self.cosmo.z_to_Chi(self._get_z(snap))

    def _window_LSST(self, z, zmin, zmax):
        return self.cosmo.gal_window_z(z, zmin=zmin, zmax=zmax)
    
    def _window_LSST_mu(self, z):
        # Need to divide by H(z) to convert window into function of z and not chi
        return self.cosmo.gal_lens_window_matter(self.cosmo.z_to_Chi(z), self.cosmo.get_chi_star())/self.cosmo.get_hubble(z)

    def _window_Planck_cib(self, z):
        return self.cosmo.cib_window_z(z)

    def _window_CMB(self, z):
        # Need to divide by H(z) to convert window into function of z and not chi
        return self.cosmo.cmb_lens_window_matter(self.cosmo.z_to_Chi(z), self.cosmo.get_chi_star())/self.cosmo.get_hubble(z)
    
    def _window_gal_kappa(self, snap, source_snap):
        # Need to divide by H(z) to convert window into function of z and not chi
        zmin = self._get_zmin(snap)
        zmax = self._get_zmax(snap)
        zs = np.linspace(zmin, zmax, 1000)
        dz = zs[1] - zs[0]
        wins = self.cosmo.cmb_lens_window_matter(self.cosmo.z_to_Chi(zs), self._get_chi(source_snap))/self.cosmo.get_hubble(zs)
        zerod_indices = np.logical_or(np.isnan(wins), np.isinf(wins))
        wins[zerod_indices] = 0
        return np.sum(wins) * dz



    def _window(self, snap, typ, gal_zmin=None, gal_zmax=None):
        zmin = self._get_zmin(snap)
        zmax = self._get_zmax(snap)
        zs = np.linspace(zmin, zmax, 1000)
        dz = zs[1] - zs[0]
        if typ == "LSST":
            return np.sum(self._window_LSST(zs, gal_zmin, gal_zmax)) * dz
        if typ == "LSST_mu":
            return np.sum(self._window_LSST_mu(zs)) * dz
        if typ == "Planck_cib":
            return np.sum(self._window_Planck_cib(zs)) * dz
        if typ == "CMB":
            wins = self._window_CMB(zs)
            zerod_indices = np.logical_or(np.isnan(wins), np.isinf(wins))
            wins[zerod_indices] = 0
            return np.sum(wins) * dz
        raise ValueError("Window type not of 'LSST' (gal), 'Planck' (CIB), or 'CMB' (CMB lensing).")
        
    def _apply_pixel_correction(self, map):
        alm = self.sht.map2alm(map)
        alm_corr = self.sht.almxfl(alm, 1/self.sht.pixwin)
        return self.sht.alm2map(alm_corr)

    def get_obs_gal_map(self, zmin=0, zmax=1100, verbose=False, pixel_corr=True, lensed=False):
        window="LSST"
        if verbose: print(f"DEMNUnii: Constructing gal map for zmin={zmin}, zmax={zmax}, window={window}, pix_cor={pixel_corr}, lensed={lensed}")
        npix = self.sht.nside2npix()
        gal = np.zeros(npix)
        snaps = self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            gal += self._window(snap, window, zmin, zmax) * self.get_density_snap(snap, lensed)
            if verbose: print('\r', end='')
        if verbose: print("")
        if pixel_corr: gal = self._apply_pixel_correction(gal)
        return gal
    
    def get_obs_mag_bias_map(self, zmin=0, zmax=1100, verbose=False, pixel_corr=True, lensed=False):
        # TODO: This assumes s=0.2 
        window="LSST_mu"
        if verbose: print(f"DEMNUnii: Constructing mag bias map for zmin={zmin}, zmax={zmax}, window={window}, pix_cor={pixel_corr}, lensed={lensed}")
        npix = self.sht.nside2npix()
        gal = np.zeros(npix)
        snaps = self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            gal += self._window(snap, window) * self.get_density_snap(snap, lensed)
            if verbose: print('\r', end='')
        if verbose: print("")
        if pixel_corr: gal = self._apply_pixel_correction(gal)
        return gal

    def get_obs_cib_map(self, zmin=0, zmax=1100, verbose=False, pixel_corr=True, lensed=False):
        window = "Planck_cib"
        if verbose: print(f"DEMNUnii: Constructing CIB map for zmin={zmin}, zmax={zmax}, window={window}, pix_cor={pixel_corr}, lensed={lensed}")
        npix = self.sht.nside2npix()
        cib = np.zeros(npix)
        snaps = self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            cib += self._window(snap, window) * self.get_density_snap(snap, lensed)
            if verbose: print('\r', end='')
        if verbose: print("")
        if pixel_corr: cib = self._apply_pixel_correction(cib)
        return cib
    
    def get_gal_kappa_map(self, source_snap, verbose=False, pixel_corr=True):
        if source_snap == 62:
            print("Returning zerod map for snap 62 (z: 0)")
            return np.zeros(self.sht.nside2npix())
        zmin = 0
        zmax = self._get_zmax(source_snap)
        if verbose: print(f"DEMNUnii: Constructing galaxy kappa map for zmin={zmin} and zmax={zmax}")
        npix = self.sht.nside2npix()
        gal_kappa = np.zeros(npix)
        snaps = self.get_snaps_z(zmin, zmax)
        assert source_snap == np.min(snaps), (source_snap, snaps)
        snaps = np.sort(snaps)[1:]
        print(f"    Source snap: {source_snap}, so using snaps {snaps[-1]} -> {snaps[0]} (zmax: {self._get_zmax(snaps[0])}) to construct kappa.")
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            gal_kappa += self.get_gal_kappa_snap(snap, source_snap)
            if verbose: print('\r', end='')
        if verbose: print("")
        if pixel_corr: gal_kappa = self._apply_pixel_correction(gal_kappa)
        return gal_kappa
    
    def get_gal_kappa_snap(self, snap, source_snap, pixel_corr=False):
        gal_kappa = self._window_gal_kappa(snap, source_snap) * self.get_density_snap(snap)
        if pixel_corr: gal_kappa = self._apply_pixel_correction(gal_kappa)
        return gal_kappa
    
    def get_kappa_map_z(self, zmin=0, zmax=1100, verbose=False, pixel_corr=True):
        window = "CMB"
        if verbose: print(f"DEMNUnii: Constructing kappa map for zmin={zmin}, zmax={zmax}, window={window}")
        npix = self.sht.nside2npix()
        kappa = np.zeros(npix)
        snaps = self.get_snaps_z(zmin, zmax)
        t0 = datetime.datetime.now()
        for iii, snap in enumerate(snaps):
            if verbose: print(f"    [{str(datetime.datetime.now() - t0)[:-7]}] Snap: {snap} ({iii+1}/{np.size(snaps)})", end='')
            kappa += self.get_kappa_snap(snap, window)
            if verbose: print('\r', end='')
        if verbose: print("")
        if pixel_corr: kappa = self._apply_pixel_correction(kappa)
        return kappa
    
    def get_kappa_snap(self, snap, window="CMB", pixel_corr=False):
        kappa = self._window(snap, window) * self.get_density_snap(snap)
        if pixel_corr: kappa = self._apply_pixel_correction(kappa)
        return kappa

    def get_density_distro(self):
        snaps = self.get_snaps_z(0, 1100)
        distro = np.zeros(np.size(snaps))
        for iii, snap in enumerate(snaps):
            distro[iii] = np.sum(self.get_particle_snap(snap))
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
    
    @staticmethod
    def gaussian_smooth_1d(data, sigma=1, nsigma=1):
        win = signal.gaussian(nsigma * sigma, std=sigma)
        res = np.convolve(data, win, mode='same')
        return res / np.convolve(np.ones(data.shape), win, mode='same')
    
    def get_PK(self, typ="total"):
        power = Powerspectra(cosmology=self.cosmo)
        typ_name = "" if typ == "total" else "_" + typ 
        h = (self.cosmo._pars.H0 / 100)
        class PKInterpolator(RectBivariateSpline):
            def P(self, z, kh, grid=None):
                if grid is None:
                    grid = not np.isscalar(z) and not np.isscalar(kh)
                else:
                    return self(z, np.log(kh), grid=grid)
                
        def _get_k(snap):
            zeros="0" if snap>9 else "00"
            ks_ = np.loadtxt(f"{self.data_dir}/powerspectra_LCDM_2Gp_2048_PlTab5_binned_II/ps{typ_name}_{zeros}{snap}.txt", usecols=0, skiprows=1)
            return ks_ * h

        def _get_pk(snap):
            zeros="0" if snap>9 else "00"
            pk_ = np.loadtxt(f"{self.data_dir}/powerspectra_LCDM_2Gp_2048_PlTab5_binned_II/ps{typ_name}_{zeros}{snap}.txt", usecols=2, skiprows=1)
            return pk_/h**3
        
        Nsnaps = 63
        snaps = np.arange(Nsnaps)
        # ks = _get_k(snaps[-1])
        # zs = np.array([self._get_z(snap) for snap in snaps[::-1]])
        # ps = np.zeros((Nsnaps, np.size(ks)))
        # kmin = 2e-3
        # kmax = 1
        # for iii, snap in enumerate(snaps[::-1]):
        #     ks_ = _get_k(snap)
        #     ps_ = _get_pk(snap)
        #     ps_[ks_<kmin] = power.get_matter_ps("matter", self._get_z(snap), ks_[ks_<kmin])
        #     ps_[ks_>kmax] = power.get_matter_ps("matter", self._get_z(snap), ks_[ks_>kmax])
        #     ps[iii,:] = InterpolatedUnivariateSpline(ks_, ps_)(ks)

        ks = np.geomspace(1e-4,1e3,1000)
        zs = np.array([self._get_z(snap) for snap in snaps[::-1]])
        ps = np.zeros((Nsnaps, np.size(ks)))
        # Will use Mead in interpolator over the edges (don't think it makes much difference)
        kmin = 2e-3
        kmax = 1
        for iii, snap in enumerate(snaps[::-1]):
            ks_ = _get_k(snap)
            ps_ = _get_pk(snap)
            # ps_ = self.gaussian_smooth_1d(ps_)    # Smoothing empirical Pk
            ps[iii,:] = power.get_matter_ps("matter", self._get_z(snap), ks)
            ks_overlap_idx = np.logical_and(ks>kmin,ks<kmax)
            ks_overlap = ks[ks_overlap_idx]
            ps[iii,np.logical_and(ks>kmin,ks<kmax)] = InterpolatedUnivariateSpline(ks_, ps_)(ks_overlap)


        PK = PKInterpolator(zs, np.log(ks), ps)
        PK.kmin = np.min(ks)
        PK.kmax = np.max(ks)
        PK.islog = False
        PK.logsign = 1
        PK.zmin = np.min(zs)
        PK.zmax = np.max(zs)
        return PK
        

        
        
        
