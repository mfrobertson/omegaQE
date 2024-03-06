import numpy as np
import omegaqe
import omegaqe.tools as tools
import sys
from omegaqe.modecoupling import Modecoupling
from DEMNUnii.demnunii import Demnunii

dm = Demnunii(nthreads=8)
cache_dir = omegaqe.CACHE_DIR

def M_matrix(mode, ells, star, typ):
    Nells = np.size(ells)
    M = np.ones((Nells, Nells))
    print(f"{0}/{Nells}", end='')
    for iii, ell in enumerate(ells):
        print('\r', end='')
        print(f"{iii}/{Nells}", end='')
        M[iii, :] = mode.components(np.ones(Nells) * ell, ells, typ=typ, star=star)
    return M


def save(ells, M, star, typ):
    sep = tools.getFileSep()
    folder = cache_dir + sep + "_M_dm" + sep + typ + sep + f"{ellmax}_{Nells}"
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
    matter_PK = dm.get_PK()
    mode.matter_PK = matter_PK
    mode._powerspectra.matter_PK = matter_PK
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
