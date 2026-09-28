import sys
import os; sys.path.insert(0, os.environ.get('LATTICE_ESTIMATOR', 'lattice-estimator'))
from estimator import *
for (n, logsig) in [(1536, log(3.2*2**24,2)), (2048, log(3.2*2**24,2))]:
    params = LWE.Parameters(n=n, q=2**64, Xs=ND.Uniform(0,1,n), Xe=ND.DiscreteGaussian(2**float(logsig)), m=2*n, tag="RotaFace-n%d"%n)
    print(params)
    r = LWE.estimate.rough(params)
    print("ROUGH", n, r)
    sys.stdout.flush()
