import numpy as np
import tools
import sys
import os
from scipy.interpolate import InterpolatedUnivariateSpline

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from qe import QE

def save(N0, fields, gmv, resp_ps, exp):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sep = tools.getFileSep()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if gmv:
        gmv_str = "gmv"
    else:
        gmv_str = "single"
    folder = current_folder + sep + "_N0" + sep + exp + sep + gmv_str
    filename_N0 = f"N0_{fields}_{resp_ps}.npy"
    tools.save_array(folder, filename_N0, N0)

def main(exp, fields, gmv, resp_ps, Nell):
    qe = QE(exp=exp)
    samp1 = np.arange(2, 40, 1)
    samp2 = np.logspace(1, 3, Nell-38) * 4
    Ls = np.concatenate((samp1, samp2))
    if gmv:
        N0_phi = qe.gmv_normalisation(Ls, curl=False, fields=fields, resp_ps=resp_ps, Lmin=2)
        N0_curl = qe.gmv_normalisation(Ls, curl=True, fields=fields, resp_ps=resp_ps, Lmin=2)
    else:
        N0_phi = qe.normalisation(fields, Ls, curl=False, resp_ps=resp_ps, Lmin=2)
        N0_curl = qe.normalisation(fields, Ls, curl=True, resp_ps=resp_ps, Lmin=2)
    N0_phi_spline = InterpolatedUnivariateSpline(Ls, N0_phi)
    N0_curl_spline = InterpolatedUnivariateSpline(Ls, N0_curl)
    sample_Ls = np.arange(2, 4001)
    N0 = (N0_phi_spline(sample_Ls), N0_curl_spline(sample_Ls))
    save(N0, fields, gmv, resp_ps, exp)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        raise ValueError("Must supply arguments: exp fields gmv resp_ps Nell")
    exp = str(args[0])
    fields = str(args[1])
    gmv = tools.parse_boolean(args[2])
    resp_ps = str(args[3])
    Nell = int(args[4])
    main(exp, fields, gmv, resp_ps, Nell)
