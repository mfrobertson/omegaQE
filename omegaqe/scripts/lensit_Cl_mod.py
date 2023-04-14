import numpy as np
import sys
from omegaqe.powerspectra import Powerspectra


def _get_Cl_pp(ells):
    return Powerspectra().get_kappa_ps(ells) * 4 / (2*np.pi)

def _get_Cl_oo(ells):
    return np.zeros(np.size(ells))


def main(path_to_lensit):
    cols = np.loadtxt(f'{path_to_lensit}/lensit/data/cls/fiducial_flatsky_lenspotentialCls.dat').transpose()
    ells = cols[0]
    Cl_pp = _get_Cl_pp(ells)
    cols[5] = Cl_pp
    # np.savetxt(f'my_fiducial_flatsky_lenspotentialCls.dat', cols.transpose(), fmt='%.5e')    #File to be added to /lensit/data/cls/
    col = np.loadtxt(f'{path_to_lensit}/lensit/data/cls/fiducial_fieldrotationCls.dat').transpose()
    ells = np.arange(np.size(col))
    # col = _get_Cl_oo(ells)
    col *=1e3
    np.savetxt(f'my_fiducial_fieldrotationCls.dat', col.transpose(), fmt='%.5e')    #File to be added to /lensit/data/cls/



if __name__ == '__main__':
    # NOTE: must delete all cached lensit data to implement the change
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError("Must supply arguments: path_to_lensit")
    main(str(args[0]))    # This changing the lensit input phi Cl to be zeroth order limber
