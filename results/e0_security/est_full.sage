import sys
import os; sys.path.insert(0, os.environ.get('LATTICE_ESTIMATOR', 'lattice-estimator'))
from estimator import *
for n in (1536, 2048):
    params = LWE.Parameters(n=n, q=2**64, Xs=ND.Uniform(0,1,n), Xe=ND.DiscreteGaussian(3.2*2**24), m=2*n, tag="n%d"%n)
    r = LWE.estimate(params, jobs=4)
    print("FULL", n, {k: float(log(v["rop"],2)) for k,v in r.items()})
    sys.stdout.flush()
