import numpy as np
import sys
import os
from omegaqe.tools import mpi, parse_boolean
from omegaqe.noise import Noise
from fullsky_sims.agora import Agora
from scipy.constants import Planck, physical_constants


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

# SPT-3G (for testing and comparing with AGORA paper)
T_ell_knee_spt = [1200, 2200, 2300]
P_ell_knee_spt = [300, 300, 300]
T_alpha_spt = [-3, -4, -4]
P_alpha_spt = [-1, -1, -1]
n_lev_T_spt = [3.0, 2.0, 9.0]
n_lev_P_spt = [4.2, 2.8, 12.4]
beam_spt = [1.6, 1.2, 1.1]


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

def _get_b(nu):
    # TSZmixing vector eq8 in 1006.5599
    T = 2.7255
    h = Planck
    k_B = physical_constants["Boltzmann constant"][0]
    x = h*nu/(k_B*T)
    return x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

def _get_constrained_w(c_inv):
    a = np.ones(np.shape(c_inv)[-1]).T
    _freqs = freqs + [545] if np.size(freqs) != np.size(a) else freqs   #tmp solution to if planck=True
    b = np.array([_get_b(freq*1e9) for freq in _freqs]).T
    bb = np.matmul(np.matmul(b.T, c_inv), b)[:,None]
    aa = np.matmul(np.matmul(a.T, c_inv), a)[:,None]
    ab = np.matmul(np.matmul(a.T, c_inv), b)[:,None]
    c_inv_a = np.matmul(c_inv, a)
    c_inv_b = np.matmul(c_inv, b)
    return ((bb*c_inv_a) - (ab*c_inv_b))/((aa*bb) - (ab)**2)

def _get_w(c_inv, constrained=False):
    if constrained:
        return _get_constrained_w(c_inv)
    a = np.ones(np.shape(c_inv)[-1]).T
    num = np.matmul(c_inv, a)
    denom = np.matmul(np.matmul(a.T, c_inv),a)[:,None]
    return num/denom

def _fg_maps(tsz, ksz, cib, rad, planck, point, cluster):
    Nfreqs = len(freqs)
    if planck: Nfreqs += 1
    Ts_fg = np.empty((Nfreqs, ag.sht.nside2npix(ag.nside)))
    Qs_fg = np.empty((Nfreqs, ag.sht.nside2npix(ag.nside)))
    Us_fg = np.empty((Nfreqs, ag.sht.nside2npix(ag.nside)))
    for iii, freq in enumerate(freqs):
        T_fg, Q_fg, U_fg = ag.create_fg_maps(freq, tsz, ksz, cib, rad, False, point, cluster)
        Ts_fg[iii,:] = T_fg
        Qs_fg[iii,:] = Q_fg
        Us_fg[iii,:] = U_fg
    if planck:
        T_pl, Q_pl, U_pl = ag.create_fg_maps(545, tsz, ksz, cib, False, False, point, cluster)
        Ts_fg[-1,:] = T_pl
        Qs_fg[-1,:] = Q_pl
        Us_fg[-1,:] = U_pl
    return Ts_fg, Qs_fg, Us_fg

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

def _noise_cls(exp, planck):
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
        for iii, nu in enumerate(freqs):
            N_Ts[iii,:] = _noise_cl(None, T_ell_knee_spt[iii], T_alpha_spt[iii], beam_spt[iii], n_lev_T_spt[iii])
            N_Ps[iii,:] = _noise_cl(None, P_ell_knee_spt[iii], P_alpha_spt[iii], beam_spt[iii], n_lev_P_spt[iii])
        return N_Ts * scaling, N_Ps * scaling
    else:
        raise ValueError(f"Argument exp: {exp} not expected. Try SO_base or SO_goal or S4_base.")
    Nfreqs = len(freqs)
    if planck: Nfreqs += 1
    N_Ts = np.empty((Nfreqs, ag.Lmax_map + 1))
    N_Ps = np.empty((Nfreqs, ag.Lmax_map + 1))
    for iii, nu in enumerate(freqs):
        N_Ts[iii,:] = _noise_cl(Nred_Ts[iii], T_ell_knee, T_alpha, beams[iii], n_lev_array[iii])
        N_Ps[iii,:] = _noise_cl(None, P_ell_knee, P_alpha, beams[iii], n_lev_array[iii]*np.sqrt(2))
    if planck:
        N_Ts[-1,:] = _noise_cl(0, -1, -1, 7, 35)
        N_Ps[-1,:] = _noise_cl(0, -1, -1, 7, 35*np.sqrt(2))
        N_Ts[-1,0] = N_Ps[-1,0] = np.inf
    return N_Ts * scaling, N_Ps * scaling

def setup_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

def save_ws(exp, w_T, w_E, w_B, constrained, planck, point, cluster):
    dir_name = "HILC_weights_new"
    if constrained:
        dir_name += "_constrained"
    if planck:
        dir_name += "_planck"
    mask = ""
    if point:
        mask += "_point"
    if cluster:
        mask += "_cluster"
    full_dir = f"{ag.cache_dir}/_{dir_name}/{exp}/{mask}"
    setup_dir(full_dir)
    filename = "weights"
    for nu in freqs:
        filename += f"_{nu}"
    np.save(f"{full_dir}/{filename}_T", w_T)
    np.save(f"{full_dir}/{filename}_E", w_E)
    np.save(f"{full_dir}/{filename}_B", w_B)

def main(exp, tsz, ksz, cib, rad, constrained, planck, point, cluster, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f" tsz: {tsz}, ksz: {ksz}, cib: {cib}, rad: {rad}, constrained: {constrained}, planck: {planck}, point: {point}, cluster: {cluster}, nthreads: {nthreads}", 0, _id)

    global ag, noise
    ag = Agora(nthreads=nthreads)
    noise = Noise(cosmology=ag.cosmo)
    noise.full_sky = True


    Ts, Qs, Us = _fg_maps(tsz, ksz, cib, rad, planck, point, cluster)

    N_Ts, N_Ps = _noise_cls(exp, planck)

    C_T_inv = _c_inv_T(Ts, N_Ts)
    C_E_inv, C_B_inv = _c_inv_P(Qs, Us, N_Ps)

    w_T = _get_w(C_T_inv, constrained)
    w_E = _get_w(C_E_inv, constrained)
    w_B = _get_w(C_B_inv, constrained)

    save_ws(exp, w_T, w_E, w_B, constrained, planck, point, cluster)

    
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 11:
        raise ValueError(
            "Must supply arguments: exp tsz ksz cib rad constrained planck point cluster nthreads _id")
    exp = str(args[0])
    tsz = parse_boolean(args[1])
    ksz = parse_boolean(args[2])
    cib = parse_boolean(args[3])
    rad = parse_boolean(args[4])
    constrained = parse_boolean(args[5])
    planck = parse_boolean(args[6])
    point_mask = parse_boolean(args[7])
    cluster_mask = parse_boolean(args[8])
    nthreads  = int(args[9])
    _id = str(args[10])
    main(exp, tsz, ksz, cib, rad, constrained, planck, point_mask, cluster_mask, nthreads, _id)