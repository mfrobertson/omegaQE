import numpy as np
import os
import sys

# Switching path to parent directory to access and run modecoupling
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from powerspectra import Powerspectra


def _get_Cl_pp(ells):
    return Powerspectra().get_kappa_ps(ells) * 4 / (2*np.pi)

def _get_Cl_oo(ells):
    return np.zeros(np.size(ells))


def main():
    path_to_lensit = f'../../../Lensit'
    cols = np.loadtxt(f'{path_to_lensit}/lensit/data/cls/fiducial_flatsky_lenspotentialCls.dat').transpose()
    ells = cols[0]
    Cl_pp = _get_Cl_pp(ells)
    cols[5] = Cl_pp
    # np.savetxt(f'my_fiducial_flatsky_lenspotentialCls.dat', cols.transpose(), fmt='%.5e')    #File to be added to /lensit/data/cls/
    cols = np.loadtxt(f'{path_to_lensit}/lensit/data/cls/fiducial_fieldrotationCls.dat').transpose()
    ells = np.arange(np.size(cols[0]))
    cols[0] = _get_Cl_oo(ells)
    np.savetxt(f'my_fiducial_fieldrotationCls.dat', cols.transpose(), fmt='%.5e')    #File to be added to /lensit/data/cls/



if __name__ == '__main__':
    # NOTE: must delete all cached data to implement the change
    main()    # This changing the lensit input phi Cl to be zeroth order limber
