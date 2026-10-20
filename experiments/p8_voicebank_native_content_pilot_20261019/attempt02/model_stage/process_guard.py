"""Native local-query grammar. Audit only; never rewrite calls or run probes."""
from pathlib import Path
import os,re,shutil

def require(ok,message):
    if not ok:raise RuntimeError(message)
def allow_local_build(executable,argv,cwd,config):
    require(isinstance(argv,(tuple,list)) and argv,'Expected native query argv')
    command=os.fsdecode(argv[0]);resolved=shutil.which(command) if not Path(command).is_absolute() else command
    if resolved is None and command=='file':
        flags=[os.fsdecode(v) for v in argv[1:]];require(len(flags)==2 and flags[0]=='-b' and str(Path(flags[-1]).resolve()) in config['query_readonly_files'],'Unknown absent file query');return dict(executable=None,original_executable=os.fsdecode(executable),argv=list(argv),cwd=None if cwd is None else str(cwd),kind='platform_file_absent_native_OSError')
    require(resolved is not None,'Unknown local executable')
    resolved=str(Path(resolved).resolve());require(resolved in config['local_build_executables'],'Unapproved native executable '+resolved)
    flags=[os.fsdecode(v) for v in argv[1:]];name=Path(resolved).name;cwd_resolved=None if cwd is None else str(Path(cwd).resolve());kind=None
    if name in ['ldconfig','ldconfig.real']:
        require(flags==['-p'],'Only read-only ldconfig -p');kind='ctypes_ldconfig_readonly'
    elif name=='uname':
        require(flags==['-p'],'Only platform uname -p');kind='platform_processor'
    elif name=='file':
        require(len(flags)==3 and flags[:2]==['-b','-L'] or len(flags)==2 and flags[:1]==['-b'],'Only platform file query')
        target=Path(flags[-1]).resolve();require(str(target) in config['query_readonly_files'],'Unfrozen platform file target');kind='platform_architecture'
    elif resolved==config['python_executable_resolved']:
        require(len(flags)==4 and flags[:2]==['-m','cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess'] and flags[2] in ['canary','load'] and flags[3] in config['cuda_query_libnames'],'Only static CUDA dynamic library query')
        require(cwd_resolved==config['cuda_query_cwd'],'Unexpected CUDA query cwd');kind='cuda_pathfinder_'+flags[2]
    elif name=='ninja':
        if flags==['--version']:kind='native_ninja_version'
        else:
            require(flags==['-v'] or len(flags)==3 and flags[:2]==['-v','-j'] and flags[2].isdigit() and int(flags[2])>0,'Unexpected native Ninja flags')
            require(cwd_resolved in config['native_build_directories'],'Unexpected Ninja build directory');kind='native_cached_build'
    else:
        require(flags in [['-v'],['--version'],['-dumpfullversion','-dumpversion']],'Only static compiler-version grammar');kind='native_compiler_version'
    return dict(executable=resolved,original_executable=os.fsdecode(executable),argv=[os.fsdecode(v) for v in argv],cwd=cwd_resolved,kind=kind)
