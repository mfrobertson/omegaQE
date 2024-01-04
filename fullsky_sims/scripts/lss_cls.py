import fullsky_sims
from fullsky_sims.fields import Fields
import numpy as np


def main():
    nbins = 150
    fields = Fields("SO_base", use_lss_cache=True, use_cmb_cache=True, nthreads=10)
    Cl_kk = fields.get_Cl("kk")
    Cl_kk = fields.nbody.sht.smoothed_cl(Cl_kk, nbins, False)
    Cl_kg = fields.get_Cl("kg", smoothing_nbins=nbins)
    Cl_kI = fields.get_Cl("kI", smoothing_nbins=nbins)
    np.save(f"{fields.nbody.cache_dir}/_lss_cls/cl_kk.npy", Cl_kk)
    np.save(f"{fields.nbody.cache_dir}/_lss_cls/cl_gk.npy", Cl_kg)
    np.save(f"{fields.nbody.cache_dir}/_lss_cls/cl_Ik.npy", Cl_kI)


if __name__ == '__main__':
    main()
