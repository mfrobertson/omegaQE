import numpy as np
import os
import sys

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from powerspectra import Powerspectra


def _get_Cl_pp(ells):
    return Powerspectra().get_kappa_ps(ells) * 4 / (2*np.pi)


def main():
    path_to_lensit = f'../../../Lensit'
    cols = np.loadtxt(f'{path_to_lensit}/lensit/data/cls/fiducial_flatsky_lenspotentialCls.dat').transpose()
    ells = cols[0]
    Cl_pp = _get_Cl_pp(ells)
    cols[5] = Cl_pp
    #np.savetxt(f'my_fiducial_flatsky_lenspotentialCls.dat', cols.transpose(), fmt='%.5e')    #File to be added to /lensit/data/cls/fiducial_flatsky_lenspotentialCls.dat


if __name__ == '__main__':
    main()
