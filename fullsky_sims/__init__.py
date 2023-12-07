from fullsky_sims.demnunii import Demnunii
from fullsky_sims.agora import Agora

def wrapper_class(nbody, nthreads):
    if nbody.lower() == "demnunii":
        return Demnunii(nthreads)
    if nbody.lower() == "agora":
        return Agora(nthreads)
    raise ValueError(f"Nbody sim {nbody} not recognised.")