import os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from independent_readout import representation,evaluate,compare
from cache_io import execute
if __name__=='__main__':
    if sys.argv[1:]!=['--validate-after-completed-run']:raise RuntimeError('Explicit independent validation invocation')
    sys.exit(execute('validation',representation,evaluate,compare))
