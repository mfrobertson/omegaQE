import numpy as np
import tools
import sys
import os

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from fisher import Fisher

def save(ells, F_L, typs, exp):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    folder = current_folder + sep + "_F_L" + sep + exp + sep + typs + sep + f"{Nells}"
    filename_F_L = "F_L.npy"
    filename_ells = "ells.npy"
    tools.save_array(folder, filename_F_L, F_L)
    tools.save_array(folder, filename_ells, ells)

def get_noise_args(exp):
    if exp == "SO":
        return "_N0/N0_my_SO_14_14_TQU.npy", 0, True
    elif exp == "S4":
        return "_N0/N0_my_S4_14_14_TQU.npy", 0, True

def main(Nells, typs, exp):
    file, offset, ell_facs = get_noise_args(exp)
    fisher = Fisher(file, N0_offset=offset, N0_ell_factors=ell_facs)
    Ls1 = np.arange(3, 40, 1)
    Ls2 = np.logspace(1, 3, Nells-37) * 4
    ells = np.concatenate((Ls1, Ls2))
    _, F_L = fisher.get_F_L(typs, Ls=ells, Ntheta=100, nu=353e9, return_C_inv=False)
    save(ells, F_L, typs, exp)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        raise ValueError("Must supply arguments: Nells typs exp")
    Nells = int(args[0])
    typs = str(args[1])
    exp = str(args[2])
    main(Nells, typs, exp)
