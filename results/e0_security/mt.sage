import sys
import os; sys.path.insert(0, os.environ.get('LATTICE_ESTIMATOR', 'lattice-estimator'))
from estimator import *
for (n, logq, sig) in [(896, 16, 3.2), (832, 16, 3.2)]:
    P = LWE.Parameters(n=n, q=2**logq, Xs=ND.Uniform(0,1,n), Xe=ND.DiscreteGaussian(sig), m=2*n)
    r = LWE.estimate.rough(P, quiet=True)
    print("ROUGH n=%d logq=%d sigma=%.1f min=%.1f" % (n, logq, sig, min(float(log(v["rop"],2)) for v in r.values())), flush=True)
