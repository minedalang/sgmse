"""Independent business reconstruction with shared disclosed lifecycle/IO only."""
import os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from independent_integrity import independent_pair,compare
from runtime import execute
if __name__=='__main__':
    if sys.argv[1:]!=['--validate-after-completed-run']:raise RuntimeError('Explicit validation invocation required')
    sys.exit(execute('validation',independent_pair,compare))
