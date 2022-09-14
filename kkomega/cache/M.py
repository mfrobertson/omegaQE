import numpy as np
import tools
import sys
import os

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from modecoupling import Modecoupling

def M_matrix(mode, ells, star, typ):
    Nells = np.size(ells)
    M = np.ones((Nells, Nells))
    for iii, ell in enumerate(ells):
        M[iii, :] = mode.components(np.ones(Nells) * ell, ells, typ=typ, star=star)
    return M


def save(ells, M, star, typ):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    folder = current_folder + sep + "_M" + sep + typ + sep + f"{ellmax}_{Nells}"
    if star:
        folder += "_s"
    filename_M = "M.npy"
    filename_ells = "ells.npy"
    print(f"Saving {filename_M} at {folder}")
    tools.save_array(folder, filename_M, M)
    print(f"Saving {filename_ells} at {folder}")
    tools.save_array(folder, filename_ells, ells)


def main(ellmax, Nells, star, typ):
    mode = Modecoupling()
    ells = mode.generate_sample_ells(ellmax, Nells)
    M = M_matrix(mode, ells, star, typ)
    save(ells, M, star, typ)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 4:
        print("Must supply arguments: ellmax Nells star typ")
    ellmax = int(args[0])
    Nells = int(args[1])
    star = tools.parse_boolean(args[2])
    typ = str(args[3])
    main(ellmax, Nells, star, typ)
