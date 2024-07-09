from omegaqe.tools import getFileSep
from omegaqe.cosmology import Cosmology
from omegaqe.powerspectra import Powerspectra
import numpy as np
from fullsky_sims.spherical import Spherical
import healpy as hp
from scipy.constants import Planck, physical_constants
import copy

class Agora:

    def __init__(self, nthreads=1, downgrade=True):
        self.data_dir = "/mnt/lustre/users/astro/mr671/AGORA/"
        self.cache_dir = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/cache_ag/"
        self.sims_dir = f"{self.data_dir}/cmbsim2/"
        self.omegaqe_data = f"/mnt/lustre/users/astro/mr671/omegaQE/fullsky_sims/data_ag/"
        self.downgrade = downgrade
        self.nside_u = 8192
        self.nside = 4096
        self.Lmax_map = 5000
        self.sht = Spherical(self.nside, self.Lmax_map, nthreads=nthreads)
        self.sht_u = Spherical(self.nside_u, self.Lmax_map, nthreads=nthreads)
        self.cosmo = Cosmology("AGORA")
        self.cosmo.agora = True
        self._setup_dndz_splines()
        self.power = Powerspectra(cosmology=self.cosmo)
        self.point_mask_idx = None
        self.cluster_mask_idx = None
        self.cluster_mask_rads = None
        self.cov = None

    def _downgrade(self, map, nside_out=None):
        # Ensure no pixel window correction already applied to input map
        nside_out = self.nside if nside_out is None else nside_out
        return hp.ud_grade(map, nside_out)

    def calc_masks(self, threshold=10e-3, Nsources=60000):
        self.point_mask_idx = self._get_point_mask_indices(threshold, cib=True)
        self.cluster_mask_idx, self.cluster_mask_rads = self._get_cluster_mask_indices(Nsources)

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
        kappa_map, _, _ = self.sht_u.read_map(f"{self.data_dir}/phi/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")
        if self.downgrade:
            kappa_map = self._downgrade(kappa_map)
        elif pixel_corr: kappa_map = self._apply_pixel_correction(kappa_map)
        return kappa_map
    
    def get_omega_map(self):
        omega_map = -self.sht_u.read_map(f"{self.data_dir}/omega/raytrace16384_ip20_cmbkappa.zs1.omega_highzadded.fits")
        if self.downgrade:
            omega_map = self._downgrade(omega_map)
        return omega_map
    
    def get_omega_map_original(self, pixel_corr=True, high_z_gauss=False):
        omega_map = self.sht_u.read_map(f"{self.data_dir}/omega/raytrace16384_ip20_cmbkappa.zs1.omega.fits")
        if pixel_corr: omega_map = self._apply_pixel_correction(omega_map, down=False)
        if high_z_gauss:
            zlim = 8.6251
            print(f"Creating new Gaussian realization for omega at z>{zlim}")
            import omegaqe.postborn as postborn
            from scipy.interpolate import InterpolatedUnivariateSpline
            Ls = np.geomspace(1,self.Lmax_map + 1,100)
            omega_zmin = postborn.omega_ps(Ls, zmin=zlim, powerspectra=self.power)
            cl_w_add = InterpolatedUnivariateSpline(Ls, omega_zmin)(np.arange(self.Lmax_map + 1))
            omega_map += self.sht_u.synfast(cl_w_add)
        return -omega_map

    def get_gal_bin_map(self, bin=1):
        return self.sht_u.read_map(f"{self.data_dir}/gal/agora_biaseddensity_lsst_y1_lens_zbin{bin}_fullsky.fits")
    
    def _apply_pixel_correction(self, map, down=None):
        down = self.downgrade if down is None else down
        sht = self.sht if down else self.sht_u
        alm = sht.map2alm(map)
        alm_corr = sht.almxfl(alm, 1/sht.pixwin)
        return sht.alm2map(alm_corr)
    
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
        if self.downgrade:
            gal_map = self._downgrade(gal_map)
        if pixel_corr: gal_map = self._apply_pixel_correction(gal_map)
        return gal_map
    
    def get_obs_cib_map(self, nu=353, pixel_corr=True, lensed=True, verbose=False, muK=False, point_mask=False, downgrade=True):
        if not lensed:
            raise ValueError("AGORA products are all lensed.")
        if nu == 95:
            nu = 90   #TODO: Prob should account for freq diff
            cib_map = self.sht_u.read_map(f"{self.data_dir}/cib/agora_len_mag_cibmap_act_{nu}ghz_uk.fits")
        elif nu == 150 or nu == 220:
            cib_map = self.sht_u.read_map(f"{self.data_dir}/cib/agora_len_mag_cibmap_act_{nu}ghz_uk.fits")
        else:
            cib_map = self.sht_u.read_map(f"{self.data_dir}/cib/agora_len_mag_cibmap_planck_{nu}ghz.fits")
            cib_map *= self.Jy_to_muK(nu*1e9)
        if not muK:
            cib_map /= self.Jy_to_muK(nu*1e9) * 1e6   # Converting to MJy
        if point_mask:
            mask_idx = self._get_point_mask_indices() if self.point_mask_idx is None else self.point_mask_idx
            cib_map = self._mask_point(cib_map, mask_idx)
        if self.downgrade:
            cib_map = self._downgrade(cib_map)
        if pixel_corr: cib_map = self._apply_pixel_correction(cib_map)
        return cib_map
    
    def get_gal_dndz(self, bin=1):
        zs = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=0, skiprows=2)
        dndz = np.loadtxt(f"{self.data_dir}/gal/nz_y1_lens_5bins_srd.dat", usecols=bin, skiprows=2)
        return zs, dndz
    
    def get_obs_ksz_map(self, pixel_corr=True, lensed=True, agn_T_pow=80, point_mask=False):
        ksz = self.sht_u.read_map(f"{self.data_dir}/ksz/agora_lkszNG_bahamas{agn_T_pow}_bnd_unb_1.0e+12_1.0e+18_lensed.fits")
        if point_mask:
            mask_idx = self._get_point_mask_indices() if self.point_mask_idx is None else self.point_mask_idx
            ksz = self._mask_point(ksz, mask_idx)
        if self.downgrade:
            ksz_map = self._downgrade(ksz_map)
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
        y_map = self.sht_u.read_map(f"{self.data_dir}/tsz/agora_ltszNG_bahamas{agn_T_pow}_bnd_unb_1.0e+12_1.0e+18_lensed.fits")
        if self.downgrade:
            y_map = self._downgrade(y_map)
        if pixel_corr: y_map = self._apply_pixel_correction(y_map)
        return y_map
    
    def get_obs_tsz_map(self, nu, pixel_corr=True, lensed=True, agn_T_pow=80, point_mask=False, cluster_mask=False):
        tsz = self.get_obs_y_map(False, lensed, agn_T_pow) * self.y_to_muK(nu*1e9)
        if point_mask:
            mask_idx = self._get_point_mask_indices() if self.point_mask_idx is None else self.point_mask_idx
            tsz = self._mask_point(tsz, mask_idx)
        if cluster_mask:
            masks = self._get_cluster_mask_indices() if self.cluster_mask_idx is None else (self.cluster_mask_idx, self.cluster_mask_rads)
            tsz = self._mask_cluster(tsz, masks)
        if pixel_corr: tsz = self._apply_pixel_correction(tsz)
        return tsz
    
    def get_obs_rad_map(self, nu, pixel_corr=True, lensed=True, point_mask=False):
        rad = self.sht_u.read_map(f"{self.data_dir}/radio/agora_radiomap_len_universemachine_trinity_{nu}ghz_randflux_datta2018_truncgauss.0.fits", field=0)
        rad *= self.Jy_to_muK(nu*1e9, alpha=-0.75)
        if point_mask:
            mask_idx = self._get_point_mask_indices(cib=False) if self.point_mask_idx is None else self.point_mask_idx
            rad = self._mask_point(rad, mask_idx)
        if self.downgrade:
            rad = self._downgrade(rad)
        if pixel_corr: rad = self._apply_pixel_correction(rad)
        return rad
    
                
    def get_obs_rad_maps(self, nu, pixel_corr=True, lensed=True, point_mask=False):
        # B-mode radio sources not correct (but seem unimportant anyway 1911.09466)
        T, Q, U = self.sht_u.read_map(f"{self.data_dir}/radio/agora_radiomap_len_universemachine_trinity_{nu}ghz_randflux_datta2018_truncgauss.0.fits")
        # mask_limits = self.get_mask_limits(nu)
        for iii, field in enumerate([T, Q, U]):
            if point_mask:
                mask_idx = self._get_point_mask_indices(cib=False) if self.point_mask_idx is None else self.point_mask_idx
                field = self._mask_point(field, mask_idx)
            field *= self.Jy_to_muK(nu*1e9, alpha=-0.75)
            if self.downgrade:
                field = self._downgrade(field)
            if pixel_corr: field = self._apply_pixel_correction(field)
        return T, Q, U
    
    def get_PK(self):
        return self.cosmo.get_matter_PK(typ="matter")

    def _get_point_mask_indices(self, threshold=10e-3, cib=False, cib_threshold=7e-3):
        sht = self.sht if self.downgrade else self.sht_u
        rad_nu = 95
        field = self.get_obs_rad_map(rad_nu,False)
        field /= self.Jy_to_muK(rad_nu*1e9)
        indices = np.where(np.abs(field*self.sht.nside2pixarea()) > threshold)
        if cib:
            cib_nu = 220
            field = self.get_obs_cib_map(cib_nu, False, muK=True)
            field /= self.Jy_to_muK(cib_nu*1e9)
            indices_cib = np.where(np.abs(field*sht.nside2pixarea()) > cib_threshold)
            indices_tot = np.concatenate((indices[0], indices_cib[0]))
            return np.unique(indices_tot)
        return indices
    
    def _get_cluster_mask_indices(self, Nsources=60000):
        nside = self.nside if self.downgrade else self.nside_u
        mass = np.load(f"{self.data_dir}/halocat/mass.npy")
        indices = np.argsort(mass)[::-1][:Nsources]
        ra = np.load(f"{self.data_dir}/halocat/ra.npy")[indices]
        dec = np.load(f"{self.data_dir}/halocat/dec.npy")[indices]
        rad = np.load(f"{self.data_dir}/halocat/rvir.npy")[indices]
        z = np.load(f"{self.data_dir}/halocat/z.npy")[indices]
        idx = hp.pixelfunc.ang2pix(nside, np.radians(90-dec), np.radians(ra), lonlat=False)
        h = self.cosmo._pars.H0 / 100
        chi = self.cosmo.z_to_Chi(z)*1000/h    # [kpc/h]
        theta = rad/chi   # [radians]
        return idx, theta

    def _mask_point(self, field, mask_idx, typ="median"):
        nside = self.nside if self.downgrade else self.nside_u
        median = np.median(field)
        if typ == "zerod":
            field[mask_idx] = 0
        elif typ == "median":
            field[mask_idx] = median
        elif typ == "median_nb":
            nb_idx = hp.get_all_neighbours(nside, mask_idx)
            field_copy = copy.deepcopy(field)
            for iii, idx in enumerate(mask_idx):
                field[idx] = np.median(field_copy[nb_idx[:,iii]])
        return field
    
    def _mask_cluster(self, field, masks, typ="median", rad_fac=1.5):
        nside = self.nside if self.downgrade else self.nside_u
        median = np.median(field)
        mask_idxs, mask_rads = masks
        for iii, pix in enumerate(mask_idxs):
            vec = hp.pixelfunc.pix2vec(nside, pix)
            disk_idx = hp.query_disc(nside, vec, mask_rads[iii]*rad_fac)
            if typ == "zerod":
                field[disk_idx] = 0
            elif typ == "median":
                field[disk_idx] = median
        return field

    def create_fg_maps(self, nu, tsz, ksz, cib, rad, point_mask=False, cluster_mask=False):
        nside = self.nside if self.downgrade else self.nside_u
        npix = self.sht.nside2npix(nside)
        T_fg = np.zeros(npix)
        Q_fg = np.zeros(npix)
        U_fg = np.zeros(npix)

        if point_mask:
            self.point_mask_idx = self._get_point_mask_indices(cib=True)
        if cluster_mask:
            self.cluster_mask_idx, self.cluster_mask_rads = self._get_cluster_mask_indices()
        
        if tsz:
            T_fg += self.get_obs_tsz_map(nu,  point_mask=point_mask, cluster_mask=cluster_mask)
        if ksz:
            T_fg += self.get_obs_ksz_map()
        if cib:
            T_fg += self.get_obs_cib_map(nu, muK=True, point_mask=point_mask)
        if rad:
            T_rad, Q_rad, U_rad = self.get_obs_rad_maps(nu, point_mask=True)
            T_fg += T_rad
            Q_fg += Q_rad
            U_fg += U_rad
            
        return T_fg, Q_fg, U_fg
    
    def create_gauss_fg_maps(self, nu, tsz, ksz, cib, rad, point_mask=False, cluster_mask=False, return_tracers=False, input_kappa=None):
        # TODO: if self.cov already exits it will be used regardless of whether input fg fields are the same
        def _get_cov(maps):
            N_fields = len(maps)
            fields = np.array(list(maps.keys()))
            smoothing_nbins=150
            zerod=True
            C = np.empty((self.Lmax_map, N_fields, N_fields))
            for iii in np.arange(N_fields):
                for jjj in np.arange(iii, N_fields):
                    field_i = fields[iii]
                    field_j = fields[jjj]
                    cl = self.sht.map2cl(maps[field_i], maps[field_j])
                    cl[0] = 0
                    cl_smooth = self.sht.smoothed_cl(cl, smoothing_nbins, zerod=zerod)[1:]
                    C[:, iii, jjj] = cl_smooth
                    if iii != jjj:
                        C[:, jjj, iii] = cl_smooth
            return C
        
        def _get_gauss_alm():
            return self.sht.synalm(np.ones(self.Lmax_map + 1), self.Lmax_map)

        def _get_gauss_alms(Nfields):
            alms = np.empty((self.sht.get_alm_size(), Nfields, 1), dtype="complex128")
            for iii in np.arange(Nfields):
                alms[:, iii, 0] = _get_gauss_alm()
            return alms

        def _get_L(Cov):
            N_fields = np.shape(Cov)[-1]
            L = np.linalg.cholesky(Cov)
            new_L = np.empty((self.Lmax_map + 1, N_fields, N_fields))
            new_L[1:, :, :] = L
            new_L[0, :, :] = 0.0
            return new_L

        def _matmul(L, v):
            rows = np.shape(L)[1]
            cols = np.shape(v)[2]
            res = np.empty((np.shape(v)[0], rows, cols), dtype="complex128")
            for row in np.arange(rows):
                for col in np.arange(cols):
                    for iii in np.arange(np.shape(v)[1]):
                        res[:, row, col] += self.sht.almxfl(v[:, iii, col], L[:, row, iii])
            return res

        def _get_y(input_kappa_map, cov):
            L = _get_L(cov)
            v = _get_gauss_alms(np.shape(cov)[-1])
            if input_kappa_map is not None:
                C_kappa_sqrt = L[:, 0, 0]
                C_kappa_sqrt_inv = np.zeros(np.size(C_kappa_sqrt))
                C_kappa_sqrt_inv[1:] = 1/C_kappa_sqrt[1:]
                v[:, 0, 0] = self.sht.almxfl(self.sht.map2alm(input_kappa_map), C_kappa_sqrt_inv)
            y = _matmul(L, v)
            return y
        
        if self.cov is None:
            if point_mask:
                self.point_mask_idx = self._get_point_mask_indices(cib=True)
            if cluster_mask:
                self.cluster_mask_idx, self.cluster_mask_rads = self._get_cluster_mask_indices()
            maps = {}
            maps["k"] = self.get_kappa_map()
            maps["g"] = self.get_obs_gal_map()
            maps["I"] = self.get_obs_cib_map(nu=353, muK=False)
            if tsz:
                maps["t"] = self.get_obs_tsz_map(nu, point_mask=point_mask, cluster_mask=cluster_mask)
            if ksz:
                maps["k"] = self.get_obs_ksz_map()
            if cib:
                maps["c"] = self.get_obs_cib_map(nu, muK=True, point_mask=point_mask)
            if rad:
                maps["r"] = self.get_obs_rad_map(nu, point_mask=True)
            self.cov = _get_cov(maps)
            del maps

        Nfields = np.shape(self.cov)[-1]
        kappa_map = input_kappa if input_kappa is not None else self.get_kappa_map()
        y = _get_y(kappa_map, self.cov)

        npix = self.sht.nside2npix(self.nside)
        T_fg = np.zeros(npix)
        for iii in np.arange(Nfields - 3):
            T_fg += self.sht.alm2map(copy.deepcopy(y[:, iii + 3, 0]))
        if return_tracers:
            k = self.sht.alm2map(copy.deepcopy(y[:, 0, 0]))
            g = self.sht.alm2map(copy.deepcopy(y[:, 1, 0]))
            I = self.sht.alm2map(copy.deepcopy(y[:, 2, 0]))
            return T_fg, (k, g, I)
        return T_fg