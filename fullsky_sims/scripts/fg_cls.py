from fullsky_sims.agora import Agora
import numpy as np
import os

ag = Agora(nthreads=30)

def main():
    freqs = [95,150,220]
    for freq in freqs:
        if not os.path.exists(f"{ag.cache_dir}/_fg_cls/{freq}"):
            os.makedirs(f"{ag.cache_dir}/_fg_cls/{freq}")

        T, Q, U = ag.get_obs_rad_maps(freq, point_mask=True)
        cl_rad = ag.sht.map2cl(T)
        alm_e, alm_b = ag.sht.map2alm_spin(np.array([Q, U]),2)
        cl_rad_e = ag.sht.alm2cl(alm_e)
        cl_rad_b = ag.sht.alm2cl(alm_b)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/rad_T.npy", cl_rad)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/rad_E.npy", cl_rad_e)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/rad_B.npy", cl_rad_b)

        ksz = ag.get_obs_ksz_map()
        cl_ksz = ag.sht.map2cl(ksz)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/ksz.npy", cl_ksz)

        tsz = ag.get_obs_tsz_map(freq)
        cl_tsz = ag.sht.map2cl(tsz)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/tsz.npy", cl_tsz)


        cib = ag.get_obs_cib_map(freq, muK=True)
        cl_cib = ag.sht.map2cl(cib)
        np.save(f"{ag.cache_dir}/_fg_cls/{freq}/cib.npy", cl_cib)




if __name__=="__main__":
    main()