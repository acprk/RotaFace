import sys
import os; sys.path.insert(0, os.environ.get('LATTICE_ESTIMATOR', 'lattice-estimator'))
from estimator import *
for (n, ls) in ((1536, 34.0),):
    params = LWE.Parameters(n=n, q=2**64, Xs=ND.Uniform(0,1,n), Xe=ND.DiscreteGaussian(2**ls), m=2*n, tag="n%d"%n)
    r = LWE.estimate(params, jobs=4)
    print("FULL", n, ls, {k: float(log(v["rop"],2)) for k,v in r.items()})
    sys.stdout.flush()
