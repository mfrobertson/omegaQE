from DEMNUnii.demnunii import Demnunii


def main():
    dm = Demnunii()
    kappa_map = dm.get_kappa_map()
    omega_map = dm.get_omega_map()
    gal_map = dm.get_obs_gal_map(verbose=True)
    cib_map = dm.get_obs_cib_map(verbose=True)

    dm.sht.write_map(f"{dm.cache_dir}/_maps/k2.fits", kappa_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/w2.fits", omega_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/g2.fits", gal_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/I2.fits", cib_map)

    gal_map = dm.get_obs_gal_map(verbose=True, lensed=True)
    cib_map = dm.get_obs_cib_map(verbose=True, lensed=True)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/g2_len.fits", gal_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/I2_len.fits", cib_map)


if __name__ == '__main__':
    main()
