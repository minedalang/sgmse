"""Worker-only once after Root authorization. No implicit execution on import."""
import os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from event_integrity import pair
from runtime import execute
if __name__=='__main__':
    if sys.argv[1:]!=['--execute-after-root-review']:raise RuntimeError('Explicit Worker-only invocation required')
    sys.exit(execute('main',pair))
