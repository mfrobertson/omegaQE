import numpy as np
import tools
import sys
import os

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from fisher import Fisher

def save(ells, F_L, typs, exp, gmv, fields):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    gmv_str = "gmv_str" if gmv else "single"
    current_folder = os.path.dirname(os.path.realpath(__file__))
    folder = current_folder + sep + "_F_L" + sep + exp + sep + gmv_str + sep + fields + sep + typs + sep + f"{Nells}"
    filename_F_L = "F_L.npy"
    filename_ells = "ells.npy"
    tools.save_array(folder, filename_F_L, F_L)
    tools.save_array(folder, filename_ells, ells)

def get_noise_args(exp, gmv, fields):
    sep = tools.getFileSep()
    gmv_str = "gmv_str" if gmv else "single"
    return "_N0"+sep+exp+sep+gmv_str+sep+f"N0_{fields}_gradient.npy", 2, True

def main(Nells, typs, exp, gmv, fields):
    file, offset, ell_facs = get_noise_args(exp, gmv, fields)
    fisher = Fisher(file, N0_offset=offset, N0_ell_factors=ell_facs)
    Ls1 = np.arange(3, 40, 1)
    Ls2 = np.logspace(1, 3, Nells-37) * 4
    ells = np.concatenate((Ls1, Ls2))
    _, F_L = fisher.get_F_L(typs, Ls=ells, Ntheta=100, nu=353e9, return_C_inv=False)
    save(ells, F_L, typs, exp, gmv, fields)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        raise ValueError("Must supply arguments: Nells typs exp gmv fields")
    Nells = int(args[0])
    typs = str(args[1])
    exp = str(args[2])
    gmv = tools.parse_boolean(args[3])
    fields = str(args[4])
    main(Nells, typs, exp, gmv, fields)
