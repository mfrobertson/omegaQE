import fullsky_sims


def main(nbody="DEMNUnii", nthreads=30):
    nbody = fullsky_sims.wrapper_class(nbody, nthreads)
    kappa_map = nbody.get_kappa_map(pb=True)

    nbody.sht.write_map(f"{nbody.cache_dir}/_maps/k.fits", kappa_map)

    gal_map = nbody.get_obs_gal_map(verbose=True, lensed=True)
    cib_map = nbody.get_obs_cib_map(verbose=True, lensed=True)
    nbody.sht.write_map(f"{nbody.cache_dir}/_maps/g_len.fits", gal_map)
    nbody.sht.write_map(f"{nbody.cache_dir}/_maps/I_len.fits", cib_map)


if __name__ == '__main__':
    main()
