import numpy as np
import omegaqe
import omegaqe.tools as tools
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
from omegaqe.qe import QE

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

def main(exp, fields, gmv, resp_ps, Nell, T_Lmin, T_Lmax, P_Lmin, P_Lmax):
    qe = QE(exp=exp, data_dir=data_dir)
    Ls = qe.get_log_sample_Ls(2, 5000, Nell)
    if gmv:
        N0_phi = qe.gmv_normalisation(Ls, curl=False, fields=fields, resp_ps=resp_ps, T_Lmin=T_Lmin, T_Lmax=T_Lmax, P_Lmin=P_Lmin, P_Lmax=P_Lmax)
        N0_curl = qe.gmv_normalisation(Ls, curl=True, fields=fields, resp_ps=resp_ps, T_Lmin=T_Lmin, T_Lmax=T_Lmax, P_Lmin=P_Lmin, P_Lmax=P_Lmax)
    else:
        N0_phi = qe.normalisation(fields, Ls, curl=False, resp_ps=resp_ps, T_Lmin=T_Lmin, T_Lmax=T_Lmax, P_Lmin=P_Lmin, P_Lmax=P_Lmax)
        N0_curl = qe.normalisation(fields, Ls, curl=True, resp_ps=resp_ps, T_Lmin=T_Lmin, T_Lmax=T_Lmax, P_Lmin=P_Lmin, P_Lmax=P_Lmax)
    N0_phi_spline = InterpolatedUnivariateSpline(Ls, N0_phi)
    N0_curl_spline = InterpolatedUnivariateSpline(Ls, N0_curl)
    sample_Ls = np.arange(2, 5001)
    N0 = (N0_phi_spline(sample_Ls), N0_curl_spline(sample_Ls))
    save(N0, fields, gmv, resp_ps, exp, T_Lmin, T_Lmax, P_Lmin, P_Lmax)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 9:
        raise ValueError("Must supply arguments: exp fields gmv resp_ps Nell T_Lmin T_Lmax P_Lmin P_Lmax")
    exp = str(args[0])
    fields = str(args[1])
    gmv = tools.parse_boolean(args[2])
    resp_ps = str(args[3])
    Nell = int(args[4])
    T_Lmin = int(args[5])
    T_Lmax = int(args[6])
    P_Lmin = int(args[7])
    P_Lmax = int(args[8])
    main(exp, fields, gmv, resp_ps, Nell, T_Lmin, T_Lmax, P_Lmin, P_Lmax)
