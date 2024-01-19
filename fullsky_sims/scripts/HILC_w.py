import numpy as np
import sys
import os
from omegaqe.tools import mpi, parse_boolean
from omegaqe.noise import Noise
from fullsky_sims.agora import Agora


# SO noise params (1808.07445)
freqs = [95, 150, 220]
beams = [2.2,1.4,1.0]
n_lev_base = [8.0,10.0,22.0]
n_lev_goal = [5.8, 6.3, 15]
P_ell_knee = 700
P_alpha = -1.4
T_ell_knee = 1000
T_alpha = -3.5

# SO Nred calc (1808.07445)
N_years = 5
years_to_seconds = 365.25 * 86400
observing_efficiency = 0.2 * 0.85  # Obsering for 1/5th available days and only using 85% of available map
fsky = 0.4   # Required to convert to units
conversion_fac = (fsky * 4 * np.pi)/(N_years * years_to_seconds * observing_efficiency)
Nred_Ts = np.array([230, 1500, 17000]) * conversion_fac


def _c_inv_T(T_fg_maps, noise_cls):
    Nfreqs = np.shape(T_fg_maps)[0]
    assert Nfreqs == np.shape(noise_cls)[0]

    Tcmb = 2.7255
    muK_sqr_conv = (Tcmb*1e6)**2
    cl_TT = ag.cosmo.get_lens_ps("TT")[:ag.Lmax_map + 1] * muK_sqr_conv

    c_mat = np.zeros((ag.Lmax_map + 1, Nfreqs, Nfreqs))
    for iii in np.arange(Nfreqs):
        for jjj in np.arange(Nfreqs):
            c_mat[:, iii,jjj] = ag.sht.map2cl(T_fg_maps[iii], T_fg_maps[jjj]) + cl_TT
            if iii == jjj:
                c_mat[:, iii,jjj] += noise_cls[iii]
    return np.linalg.pinv(c_mat)

def _c_inv_P(Q_fg_maps, U_fg_maps, noise_cls):
    Nfreqs = np.shape(Q_fg_maps)[0]
    assert Nfreqs == np.shape(noise_cls)[0]

    Tcmb = 2.7255
    muK_sqr_conv = (Tcmb*1e6)**2
    cl_EE = ag.cosmo.get_lens_ps("EE")[:ag.Lmax_map + 1] * muK_sqr_conv
    cl_BB = ag.cosmo.get_lens_ps("BB")[:ag.Lmax_map + 1] * muK_sqr_conv

    c_mat_E = np.zeros((ag.Lmax_map + 1, Nfreqs, Nfreqs))
    c_mat_B = np.zeros((ag.Lmax_map + 1, Nfreqs, Nfreqs))
    for iii in np.arange(Nfreqs):
        for jjj in np.arange(Nfreqs):
            elm_i, blm_i = ag.sht.map2alm_spin((Q_fg_maps[iii], U_fg_maps[iii]), 2)
            elm_j, blm_j = ag.sht.map2alm_spin((Q_fg_maps[jjj], U_fg_maps[jjj]), 2)
            fg_cl_EE = ag.sht.alm2cl(elm_i, elm_j)
            fg_cl_BB = ag.sht.alm2cl(blm_i, blm_j)
            c_mat_E[:, iii,jjj] = fg_cl_EE + cl_EE
            c_mat_B[:, iii,jjj] = fg_cl_BB + cl_BB
            if iii == jjj:
                c_mat_E[:, iii,jjj] += noise_cls[iii]
                c_mat_B[:, iii,jjj] += noise_cls[iii]
    return np.linalg.pinv(c_mat_E), np.linalg.pinv(c_mat_B)

def _get_w(c_inv):
    a = np.ones(np.shape(c_inv)[-1]).T
    num = np.matmul(c_inv, a)
    denom = np.matmul(np.matmul(a.T, c_inv),a)[:,None]
    return num/denom

def _fg_maps(tsz, ksz, cib, rad):
    dir_name = f"HLIC"

    Ts = np.empty((3, ag.sht.nside2npix(ag.nside)))
    Qs = np.empty((3, ag.sht.nside2npix(ag.nside)))
    Us = np.empty((3, ag.sht.nside2npix(ag.nside)))
    for iii, nu in enumerate(freqs):
        npix = ag.sht.nside2npix(ag.nside)
        T_fg = np.zeros(npix)
        Q_fg = np.zeros(npix)
        U_fg = np.zeros(npix)

        T_tsz = ag.get_obs_tsz_map(nu)
        T_ksz = ag.get_obs_ksz_map()
        T_cib = ag.get_obs_cib_map(nu, muK=True)
        T_rad, Q_rad, U_rad = ag.get_obs_rad_maps(nu)
        
        if tsz:
            T_fg += T_tsz
            dir_name += "_tsz"
        if ksz:
            T_fg += T_ksz
            dir_name += "_ksz"
        if cib:
            T_fg += T_cib
            dir_name += "_cib"
        if rad:
            T_fg += T_rad
            Q_fg += Q_rad
            U_fg += U_rad
            dir_name += "_rad"

        Ts[iii,:] = T_fg
        Qs[iii,:] = Q_fg
        Us[iii,:] = U_fg

    full_dir = f"{ag.sims_dir}/{dir_name}"
    setup_dir(full_dir)
    return Ts, Qs, Us

def _get_N_white(beam, n_lev):
    T_cmb = 2.7255 
    return noise.get_cmb_gaussian_N("TT", deltaT=n_lev, beam=beam, ellmax=ag.Lmax_map)*(1e6*T_cmb)**2

def _noise_cl(N_red, ell_knee, alpha, beam, n_lev):
    ells = np.arange(ag.Lmax_map + 1)
    N_white = _get_N_white(beam, n_lev)
    if N_red is None:
        N_red = N_white
    else:
        arcmin_to_radians = np.pi/180/60
        beam *= arcmin_to_radians
        N_red *= np.exp(ells*(ells+1)*beam**2/(8*np.log(2)))
    return N_red*(ells/ell_knee)**alpha + N_white

def _noise_cls(exp):
    scaling = 1
    if exp == "SO_base":
        n_lev_array = n_lev_base
    elif exp == "SO_goal":
        n_lev_array = n_lev_goal
    elif exp == "S4_base":
        n_lev_array = n_lev_goal
        scaling = 0.1    # Assuming S4 is factor of 10 better than SO goal
    elif exp == "SPT-3G":   # For testing
        N_Ts = np.empty((3, ag.Lmax_map + 1))
        N_Ps = np.empty((3, ag.Lmax_map + 1))
        T_ell_knee_spt = [1200, 2200, 2300]
        P_ell_knee_spt = [300, 300, 300]
        T_alpha_spt = [-3, -4, -4]
        P_alpha_spt = [-1, -1, -1]
        n_lev_T_spt = [3.0, 2.0, 9.0]
        n_lev_P_spt = [4.2, 2.8, 12.4]
        beam_spt = [1.6, 1.2, 1.1]
        for iii, nu in enumerate(freqs):
            N_Ts[iii,:] = _noise_cl(None, T_ell_knee_spt[iii], T_alpha_spt[iii], beam_spt[iii], n_lev_T_spt[iii])
            N_Ps[iii,:] = _noise_cl(None, P_ell_knee_spt[iii], P_alpha_spt[iii], beam_spt[iii], n_lev_P_spt[iii])
        return N_Ts * scaling, N_Ps * scaling
    else:
        raise ValueError(f"Argument exp: {exp} not expected. Try SO_base or SO_goal or S4_base.")
    N_Ts = np.empty((3, ag.Lmax_map + 1))
    N_Ps = np.empty((3, ag.Lmax_map + 1))
    for iii, nu in enumerate(freqs):
        N_Ts[iii,:] = _noise_cl(Nred_Ts[iii], T_ell_knee, T_alpha, beams[iii], n_lev_array[iii])
        N_Ps[iii,:] = _noise_cl(None, P_ell_knee, P_alpha, beams[iii], n_lev_array[iii]*np.sqrt(2))
    return N_Ts * scaling, N_Ps * scaling

def setup_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

def save_ws(exp, w_T, w_E, w_B):
    full_dir = f"{ag.cache_dir}/_HILC_weights/{exp}/"
    setup_dir(full_dir)
    filename = "weights"
    for nu in freqs:
        filename += f"_{nu}"
    np.save(f"{full_dir}/{filename}_T", w_T)
    np.save(f"{full_dir}/{filename}_E", w_E)
    np.save(f"{full_dir}/{filename}_B", w_B)




def main(exp, tsz, ksz, cib, rad, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f" tsz: {tsz}, ksz: {ksz}, cib: {cib}, rad: {rad}, nthreads: {nthreads}", 0, _id)

    global ag, noise
    ag = Agora(nthreads=nthreads)
    noise = Noise(cosmology=ag.cosmo)
    noise.full_sky = True



    Ts, Qs, Us = _fg_maps(tsz, ksz, cib, rad)

    N_Ts, N_Ps = _noise_cls(exp)

    C_T_inv = _c_inv_T(Ts, N_Ts)
    C_E_inv, C_B_inv = _c_inv_P(Qs, Us, N_Ps)

    w_T = _get_w(C_T_inv)
    w_E = _get_w(C_E_inv)
    w_B = _get_w(C_B_inv)

    save_ws(exp, w_T, w_E, w_B)

    
    
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        raise ValueError(
            "Must supply arguments: exp tsz ksz cib rad nthreads _id")
    exp = str(args[0])
    tsz = parse_boolean(args[1])
    ksz = parse_boolean(args[2])
    cib = parse_boolean(args[3])
    rad = parse_boolean(args[4])
    nthreads  = int(args[5])
    _id = str(args[6])
    main(exp, tsz, ksz, cib, rad, nthreads, _id)