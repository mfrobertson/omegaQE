import numpy as np
import omegaqe
from omegaqe.tools import getFileSep
import omegaqe.postborn as pb
from scipy.interpolate import InterpolatedUnivariateSpline


def main():
    Lmax = 5000
    ells = np.geomspace(1, Lmax, 500)
    sep = getFileSep()
    C_omega = pb.omega_ps(ells, Nell_prim=4000, Ntheta=2000, zmax=99)
    C_omega_spline = InterpolatedUnivariateSpline(ells, C_omega)
    C_omega_final = np.zeros(Lmax + 1)
    C_omega_final[1:] = C_omega_spline(np.arange(1,Lmax + 1))
    np.save(f"{omegaqe.CACHE_DIR}{sep}_C_omega{sep}C_omega_99.npy", C_omega_final)
    np.save(f"{omegaqe.CACHE_DIR}{sep}_C_omega{sep}Ls_99.npy", np.arange(Lmax+1))


if __name__ == '__main__':
    main()
