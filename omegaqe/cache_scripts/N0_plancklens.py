import numpy as np
import omegaqe
import omegaqe.tools as tools
import sys
from plancklens.n0s import get_N0
from omegaqe.noise import Noise

cache_dir = omegaqe.CACHE_DIR
data_dir = omegaqe.DATA_DIR
sep = tools.getFileSep()

def save(N0, fields, gmv, resp_ps, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax):
    if gmv:
        gmv_str = "gmv"
    else:
        gmv_str = "single"
    folder = cache_dir + sep + "_N0" + sep + exp + sep + gmv_str
    filename_N0 = f"N0_{fields}_{resp_ps}_T{T_Lmin}-{T_Lmax}_P{P_Lmin}-{P_Lmax}.npy"
    print(f"Saving {filename_N0} at {folder}")
    tools.save_array(folder, filename_N0, N0)

def main(exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax):
    noise = Noise()
    Tcmb = 2.7255
    fac = (Tcmb * 1e6) ** 2
    indices = ["TT", "EE", "BB"]
    arc_to_rad = np.pi / 180 / 60
    if exp == "S4_dp":
        noise_cls = {idx.lower(): np.sqrt(noise.get_cmb_gaussian_N(idx.upper(), 0.4, 2.3,  ellmax=5000)) / arc_to_rad * np.sqrt(fac) for idx in indices}
    else:
        noise_cls = {idx.lower(): np.sqrt(noise.get_cmb_gaussian_N(idx.upper(), None, None, ellmax=5000, exp=exp)) / arc_to_rad * np.sqrt(fac) for idx in indices}
    indices = ["TT", "EE", "TE", "TB", "EB", "BB"]
    my_len_cls = {idx.lower(): fac * noise.cosmo.get_lens_ps(idx) for idx in indices}
    my_cls_grad = {idx.lower(): fac * noise.cosmo.get_grad_lens_ps(idx) for idx in indices}
    N0_dict = get_N0(0, noise_cls["tt"], noise_cls["ee"],{'t': T_Lmax, 'e': P_Lmax, 'b': P_Lmax}, T_Lmin, 5000, my_len_cls, my_cls_grad, my_len_cls, my_cls_grad,joint_TP=False)
    save((N0_dict[0]['p'][2:], N0_dict[1]['p'][2:]), "TEB", True, "gradient", exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)
    save((N0_dict[0]['p_p'][2:], N0_dict[1]['p_p'][2:]), "EB", True, "gradient", exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)
    save((N0_dict[0]['ptt'][2:], N0_dict[1]['ptt'][2:]), "TT", False, "gradient", exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        raise ValueError("Must supply arguments: exp T_Lmin T_Lmax P_Lmin P_Lmax")
    exp = str(args[0])
    T_Lmin = int(args[1])
    T_Lmax = int(args[2])
    P_Lmin = int(args[3])
    P_Lmax = int(args[4])
    main(exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)
