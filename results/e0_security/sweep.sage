import sys
import os; sys.path.insert(0, os.environ.get('LATTICE_ESTIMATOR', 'lattice-estimator'))
from estimator import *
for n in (1536, 2048):
    for ls in (25.68, 28, 30, 32, 34, 36):
        P = LWE.Parameters(n=n, q=2**64, Xs=ND.Uniform(0,1,n), Xe=ND.DiscreteGaussian(2**ls), m=2*n)
        r = LWE.estimate.rough(P, quiet=True)
        print("ROUGH n=%d log2sigma=%.2f min=%.1f" % (n, ls, min(float(log(v["rop"],2)) for v in r.values())), flush=True)
