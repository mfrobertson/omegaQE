from fullsky_sims.demnunii import Demnunii


def main(nthreads=30):
    dm = Demnunii(nthreads)
    kappa_map = dm.get_kappa_map(pb=True)
    omega_map = dm.get_omega_map()
    gal_map = dm.get_obs_gal_map(verbose=True)
    cib_map = dm.get_obs_cib_map(verbose=True)

    dm.sht.write_map(f"{dm.cache_dir}/_maps/k.fits", kappa_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/w.fits", omega_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/g.fits", gal_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/I.fits", cib_map)

    gal_map = dm.get_obs_gal_map(verbose=True, lensed=True)
    cib_map = dm.get_obs_cib_map(verbose=True, lensed=True)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/g_len.fits", gal_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/I_len.fits", cib_map)


if __name__ == '__main__':
    main()
