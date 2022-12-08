import numpy as np
import tools
import sys
import os
from plancklens import utils, n0s
import plancklens

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from noise import Noise

def save(N0, fields, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    gmv_str = "iter"
    folder = current_folder + sep + "_N0" + sep + exp + sep + gmv_str
    filename_N0 = f"N0_{fields}_lensed_T{T_Lmin}-{T_Lmax}_P{P_Lmin}-{P_Lmax}.npy"
    print(f"Saving {filename_N0} at {folder}")
    tools.save_array(folder, filename_N0, N0)

def get_qe_key(fields, gmv):
    if not gmv:
        if fields != "TT":
            raise ValueError(f"Cannot get iterative N0 for single qe field pair {fields}. Only TT.")
        return "ptt"
    elif gmv:
        if fields == "EB":
            return "p_p"
        elif fields == "TEB":
            return "p"
        else:
            raise ValueError(f"Cannot get iterative N0 for minimum variance combination of {fields}. Only EB or TEB.")

def get_N_params(exp):
    if exp[:2] == "SO":
        return 3, 3
    elif exp[:2] == "S4":
        return 1, 3
    elif exp[:2] == "HD":
        return 0.5, 0.25
    else:
        raise ValueError(f"Experiment {exp} not found.")

def get_N_dict(exp, delta_T, beam, Lmax):
    if exp == "SO_base" or exp == "S4_base":
        delta_T = None
        beam = None
    N_dict = {}
    os.chdir("../")
    for fields in ["TT", "EE", "BB"]:
        N = _noise.get_cmb_gaussian_N(fields, deltaT=delta_T, beam=beam, ellmax=Lmax, exp=exp) * (2.7255 * 1e6) ** 2
        N_dict[fields.lower()] = np.concatenate((np.array([N[0], N[0]]), N))
    os.chdir("cache/")
    return N_dict

def main(exp, fields, gmv, iters, T_Lmin, T_Lmax, P_Lmin, P_Lmax):
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    qe_key = get_qe_key(fields, gmv)
    delta_T, beam = get_N_params(exp)
    N_dict = get_N_dict(exp, delta_T, beam, np.max([T_Lmax,P_Lmax]))
    N0_iter = n0s.get_N0_iter(qe_key, delta_T, delta_T * np.sqrt(2), beam, cls_unl, {'t': T_Lmin, 'e': P_Lmin, 'b': P_Lmin}, {'t': T_Lmax, 'e': P_Lmax, 'b': P_Lmax},lmax_qlm=7000, itermax=iters, ret_delcls=False, ret_curl=True, datnoise_cls=N_dict)
    N0 = (N0_iter[0][iters], N0_iter[2][iters])
    save(N0, fields, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)


if __name__ == "__main__":
    _noise = Noise()
    args = sys.argv[1:]
    if len(args) != 8:
        raise ValueError("Must supply arguments: exp fields gmv iters T_Lmin T_Lmax P_Lmin P_Lmax")
    exp = str(args[0])
    fields = str(args[1])
    gmv = tools.parse_boolean(args[2])
    iters = int(args[3])
    T_Lmin = int(args[4])
    T_Lmax = int(args[5])
    P_Lmin = int(args[6])
    P_Lmax = int(args[7])
    main(exp, fields, gmv, iters, T_Lmin, T_Lmax, P_Lmin, P_Lmax)