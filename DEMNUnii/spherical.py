import lenspyx
from lenspyx.utils_hp import almxfl, alm2cl, synalm
import healpy as hp
import copy


class Spherical:

    def __init__(self, nside, lmax, nthreads=1):
        self.nside = nside
        self.geom_info = ('healpix', {'nside': nside})
        self.geom = lenspyx.get_geom(self.geom_info)
        self.lmax = lmax
        self.nthreads = nthreads

    def _get_nside(self, nside):
        return self.nside if nside is None else nside

    def _get_lmax(self, lmax):
        return self.lmax if lmax is None else lmax

    def _get_lmax_and_nthreads(self, lmax, nthreads):
        nthreads = self.nthreads if nthreads is None else nthreads
        lmax = self._get_lmax(lmax)
        return lmax, nthreads

    def alm2lenmap(self, alms, dlm, nthreads=None):
        nthreads = self.nthreads if nthreads is None else nthreads
        return lenspyx.alm2lenmap(alms, dlm, geometry=self.geom_info, verbose=1, epsilon=1e-10, nthreads=nthreads)

    def alm2map(self, alm, lmax=None, nthreads=None):
        lmax, nthreads = self._get_lmax_and_nthreads(lmax, nthreads)
        return self.geom.alm2map(alm, lmax=lmax, mmax=lmax, nthreads=nthreads)

    def alm2map_spin(self, alms, spin, lmax=None, nthreads=None):
        lmax, nthreads = self._get_lmax_and_nthreads(lmax, nthreads)
        return self.geom.alm2map_spin(alms, spin, lmax=lmax, mmax=lmax, nthreads=nthreads)

    def map2alm(self, map, lmax=None, nthreads=None):
        map_copy = copy.deepcopy(map)
        lmax, nthreads = self._get_lmax_and_nthreads(lmax, nthreads)
        return self.geom.map2alm(map_copy, lmax=lmax, mmax=lmax, nthreads=nthreads)

    def map2alm_spin(self, maps, spin, lmax=None, nthreads=None):
        maps_copy = copy.deepcopy(maps)
        lmax, nthreads = self._get_lmax_and_nthreads(lmax, nthreads)
        return self.geom.map2alm_spin(maps_copy, spin, lmax=lmax, mmax=lmax, nthreads=nthreads)

    @staticmethod
    def almxfl(alm, fl):
        return almxfl(alm, fl, None, False)

    @staticmethod
    def read_map(filename):
        return hp.fitsfunc.read_map(filename, dtype=float, field=None)

    @staticmethod
    def write_map(filename, map):
        return hp.fitsfunc.write_map(filename, map, dtype=float, overwrite=True)

    def alm2cl(self, alm1, alm2=None, lmax_out=None, lmax=None):
        alm2 = alm1 if alm2 is None else alm2
        lmax = self._get_lmax(lmax)
        lmax_out = lmax if lmax_out is None else lmax_out
        return alm2cl(alm1, alm2, lmax, lmax, lmax_out)

    def map2cl(self, map1, map2=None, lmax_out=None, lmax=None, nthreads=None):
        alm1 = self.map2alm(map1, lmax, nthreads)
        alm2 = self.map2alm(map2, lmax, nthreads) if map2 is not None else alm1
        return self.alm2cl(alm1, alm2, lmax_out, lmax)

    def synfast(self, Cl, lmax=None):
        lmax = self._get_lmax(lmax)
        return hp.sphtfunc.synfast(Cl, self.nside, lmax, lmax)

    def synalm(self, Cl, lmax=None):
        lmax = self._get_lmax(lmax)
        return synalm(Cl, lmax=lmax, mmax=lmax)

    def nside2npix(self, nside=None):
        nside = self._get_nside(nside)
        return hp.nside2npix(nside)

    def nside2pixarea(self, nside=None):
        nside = self._get_nside(nside)
        return hp.pixelfunc.nside2pixarea(nside)