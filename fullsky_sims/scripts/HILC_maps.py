import numpy as np
from fullsky_sims.fields import Fields
import sys
import os
from omegaqe.tools import mpi, none_or_str, parse_boolean
from fullsky_sims.agora import Agora
from copy import deepcopy


def setup_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

def _get_smoothed_weights(exp, typ, pm, cm):
    mask_str = ""
    if pm:
        mask_str += "_point"
    if cm:
        mask_str += "_cluster"
    w = np.load(f"{ag.cache_dir}/_HILC_weights_new/{exp}/{mask_str}/weights_95_150_220_{typ}.npy")
    w_shape = np.shape(w)
    w_sm = np.zeros(w_shape)
    for iii in np.arange(w_shape[1]):
        w_sm[1:,iii] = ag.sht.smoothed_cl(w[1:,iii], 100, zerod=False)
    return w_sm

def _get_Tilc(Ts, wT):
    T_ilc = None
    for iii, T in enumerate(Ts):
        T_lm = ag.sht.map2alm(T)
        T_lm = ag.sht.almxfl(T_lm, wT[:,iii])
        if T_ilc is None:
            T_ilc = T_lm
        else:
            T_ilc += T_lm
    return ag.sht.alm2map(T_ilc)

def _get_EBilc(Qs, Us, wE, wB):
    E_ilc = None
    B_ilc = None
    for iii, (Q, U) in enumerate(zip(Qs, Us)):
        E_lm, B_lm = ag.sht.map2alm_spin((Q, U), 2)
        E_lm = ag.sht.almxfl(E_lm, wE[:,iii])
        B_lm = ag.sht.almxfl(B_lm, wB[:,iii])
        if E_ilc is None:
            E_ilc = E_lm
            B_ilc = B_lm
        else:
            E_ilc += E_lm
            B_ilc += B_lm
    return ag.sht.alm2map_spin((E_ilc, B_ilc), 2)

def main(start, end, deflect_typ, exp, tsz, ksz, cib, rad, gauss, point_mask, cluster_mask, nthreads, _id):
    mpi.output("-------------------------------------", 0, _id)
    mpi.output(f"start: {start}, end: {end},  deflect_typ: {deflect_typ}, exp: {exp}, tsz: {tsz}, ksz: {ksz}, cib: {cib}, rad: {rad}, gauss: {gauss}, point_mask: {point_mask}, cluster_mask: {cluster_mask}, nthreads: {nthreads}", 0, _id)

    global ag
    ag = Agora(nthreads=nthreads)

    dir_name_ext = "_gauss" if gauss else ""    
    if tsz:
        dir_name_ext += "_tsz"
    if ksz:
        dir_name_ext += "_ksz"
    if cib:
        dir_name_ext += "_cib"
    if rad:
        dir_name_ext += "_rad"
    if point_mask:
        dir_name_ext += "_pm"
    if cluster_mask:
        dir_name_ext += "_cm"

    dir_name = f"{ag.sims_dir}/{deflect_typ}/HILC{dir_name_ext}"
    setup_dir(dir_name)

    wT = _get_smoothed_weights(exp, "T", point_mask, cluster_mask)
    wE = _get_smoothed_weights(exp, "E", point_mask, cluster_mask)
    wB = _get_smoothed_weights(exp, "B", point_mask, cluster_mask)
    freqs = [95, 150, 220]
    Ts_fg = np.empty((3, ag.sht.nside2npix(ag.nside)))
    Qs_fg = np.empty((3, ag.sht.nside2npix(ag.nside)))
    Us_fg = np.empty((3, ag.sht.nside2npix(ag.nside)))
    for iii, freq in enumerate(freqs):
        T_fg, Q_fg, U_fg = ag.create_fg_maps(freq, tsz, ksz, cib, rad, gauss, point_mask, cluster_mask)
        Ts_fg[iii,:] = T_fg
        Qs_fg[iii,:] = Q_fg
        Us_fg[iii,:] = U_fg
    for sim in np.arange(start, end):
        mpi.output(f"Sim: {sim}", 0, _id)
        
        Ts = np.empty((3, ag.sht.nside2npix(ag.nside)))
        Qs = np.empty((3, ag.sht.nside2npix(ag.nside)))
        Us = np.empty((3, ag.sht.nside2npix(ag.nside)))
        T, Q, U = ag.sht.read_map(f"{ag.sims_dir}/{deflect_typ}/TQU_{sim}.fits")
        for iii, freq in enumerate(freqs):
            Ts[iii,:] = T + deepcopy(Ts_fg[iii,:])
            Qs[iii,:] = Q + deepcopy(Qs_fg[iii,:])
            Us[iii,:] = U + deepcopy(Us_fg[iii,:])
        T_ilc = _get_Tilc(Ts, wT)
        Q_ilc, U_ilc = _get_EBilc(Qs, Us, wE, wB)
        ag.sht.write_map(f"{dir_name}/TQU_{sim}.fits", (T_ilc, Q_ilc, U_ilc))
        

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 13:
        raise ValueError(
            "Must supply arguments: start end deflect_typ exp tsz ksz cib rad gauss point_mask cluster_mask nthreads _id")
    start = int(args[0])
    end = int(args[1])
    deflect_typ = none_or_str(args[2])
    exp = str(args[3])
    tsz = parse_boolean(args[4])
    ksz = parse_boolean(args[5])
    cib = parse_boolean(args[6])
    rad = parse_boolean(args[7])
    gauss = parse_boolean(args[8])
    point_mask = parse_boolean(args[9])
    cluster_mask = parse_boolean(args[10])
    nthreads  = int(args[11])
    _id = str(args[12])
    main(start, end, deflect_typ, exp, tsz, ksz, cib, rad, gauss, point_mask, cluster_mask, nthreads, _id)
