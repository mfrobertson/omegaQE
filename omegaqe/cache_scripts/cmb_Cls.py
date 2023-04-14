import numpy as np
import omegaqe
import omegaqe.tools as tools
import sys
from omegaqe.qe import QE

cache_dir = omegaqe.CACHE_DIR

def save(C, exp, Lmax):
    sep = tools.getFileSep()
    folder = cache_dir + sep + "_Cls" + sep + exp
    filename = f"Cls_cmb_{Lmax}.npy"
    print(f"Saving {filename} in {folder}")
    tools.save_array(folder, filename, C)

def main(exp):
    qe = QE(exp=exp)
    parsed_fields_all = qe.parse_fields(includeBB=True)
    N_fields = np.size(parsed_fields_all)
    Lmax = 6000
    Ls = np.arange(Lmax + 1)
    C = np.zeros((N_fields, 3, Lmax + 1))
    for iii, field in enumerate(parsed_fields_all):
        lenCl = qe.cmb[field].lenCl_spline(Ls)
        C[iii, 0, :] = lenCl
        gradCl = qe.cmb[field].gradCl_spline(Ls)
        C[iii, 1, :] = gradCl
        N = qe.cmb[field].N_spline(Ls)
        C[iii, 2, :] = N
    save(C, exp, Lmax)





if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError("Must supply arguments: exp")
    exp = str(args[0])
    main(exp)
