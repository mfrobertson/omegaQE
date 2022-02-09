import numpy as np
import tools
import sys
import os

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from modecoupling import Modecoupling

def M_matrix(mode, ells):
    Nells = np.size(ells)
    M = np.ones((Nells, Nells))
    for iii, ell in enumerate(ells):
        M[iii, :] = mode.components(np.ones(Nells) * ell, ells)
    return M


def save(ells, M):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    folder = current_folder + sep + "_M" + sep + f"{ellmax}_{Nells}"
    filename_M = "M.npy"
    filename_ells = "ells.npy"
    tools.save_array(folder, filename_M, M)
    tools.save_array(folder, filename_ells, ells)


def main(ellmax, Nells):
    mode = Modecoupling()
    ells = mode.generate_sample_ells(ellmax, Nells)
    M = M_matrix(mode, ells)
    save(ells, M)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("Must supply arguments: ellmax Nells")
    ellmax = int(args[0])
    Nells = int(args[1])
    main(ellmax, Nells)
