from DEMNUnii.demnunii import Demnunii


def main():
    dm = Demnunii()
    kappa_map = dm.get_kappa_map()
    omega_map = dm.get_omega_map()
    gal_map = dm.get_obs_gal_map(verbose=True)
    cib_map = dm.get_obs_cib_map(verbose=True)

    # hp.write_map(f"{dm.cache_dir}/_maps/k.fits", kappa_map, dtype=float, overwrite=True)
    # hp.write_map(f"{dm.cache_dir}/_maps/w.fits", omega_map, dtype=float, overwrite=True)
    # hp.write_map(f"{dm.cache_dir}/_maps/g.fits", gal_map, dtype=float, overwrite=True)
    # hp.write_map(f"{dm.cache_dir}/_maps/I.fits", cib_map, dtype=float, overwrite=True)

    dm.sht.write_map(f"{dm.cache_dir}/_maps/k.fits", kappa_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/w.fits", omega_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/g.fits", gal_map)
    dm.sht.write_map(f"{dm.cache_dir}/_maps/I.fits", cib_map)


if __name__ == '__main__':
    main()
