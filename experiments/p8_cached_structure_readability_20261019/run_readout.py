import os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from readout import representation,evaluate
from cache_io import execute
if __name__=='__main__':
    if sys.argv[1:]!=['--execute-after-root-review']:raise RuntimeError('Worker-only explicit run')
    sys.exit(execute('main',representation,evaluate))
