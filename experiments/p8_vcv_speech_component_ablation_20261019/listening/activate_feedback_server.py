"""One-time reversible activation and targeted static compatibility validation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYTHON = '/data/coding/sgmse/env/bin/python'
OLD = [PYTHON, '-m', 'http.server', '8766', '--bind', '127.0.0.1', '--directory', str(ROOT)]
PID = 15841

def launch(args, name):
    log = (HERE/name).open('ab')
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)

def ready(proc):
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError('server exited')
        try:
            urlopen('http://127.0.0.1:8766/', timeout=.5).close()
            return
        except OSError:
            time.sleep(.05)
    raise RuntimeError('server readiness failed')

if __name__ == '__main__':
    actual=Path(f'/proc/{PID}/cmdline').read_bytes().decode().split('\0')[:-1]
    assert actual == OLD, actual
    paths=[HERE/'index.html', HERE/'feedback.js', *HERE.glob('*.wav'), ROOT/'p8_10720_content_identity_audit_20261018/listening.html', ROOT/'p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_clean.wav', ROOT/'p8_q_vcv_lowsnr_challenge_20261018/listening.html', ROOT/'p8_q_vcv_lowsnr_challenge_20261018/outputs/audio/N1_snr-6/m2ada1/clean/listening.wav']
    assert all(path.is_file() for path in paths)
    os.kill(PID,signal.SIGTERM)
    time.sleep(.2)
    new=launch([PYTHON,str(HERE/'feedback_server.py'),'--port','8766'],'feedback_server.log')
    try:
        ready(new)
        def verify(p):
            data=p.read_bytes();relative=p.relative_to(ROOT).as_posix()
            fetched=urlopen('http://127.0.0.1:8766/'+quote(relative),timeout=30).read()
            assert fetched==data,relative
            head=urlopen(Request('http://127.0.0.1:8766/'+quote(relative),method='HEAD'),timeout=10)
            assert head.status==200 and int(head.headers['Content-Length'])==len(data)
            return {'path':relative,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        with ThreadPoolExecutor(max_workers=12) as pool:
            verified=list(pool.map(verify,paths))
        endpoint='http://127.0.0.1:8766/'+HERE.parent.name+'/listening/api/feedback?session_id='+str(uuid.uuid4())
        assert json.loads(urlopen(endpoint,timeout=5).read())=={'ok':True,'record':None}
        wav=next(HERE.glob('*.wav'))
        ranged=urlopen(Request('http://127.0.0.1:8766/'+wav.relative_to(ROOT).as_posix(),headers={'Range':'bytes=0-10'}))
        assert ranged.status==200 and ranged.read()==wav.read_bytes()
        try:
            urlopen('http://127.0.0.1:8766/definitely_missing_feedback_test_20261019')
            raise AssertionError('expected404')
        except HTTPError as error:
            assert error.code==404
        result={'range_semantics':'unchanged SimpleHTTPRequestHandler200/fullbody','head_passed':True,'missing_static404':True,'no_live_test_feedback_submitted':True,'status':'passed','old_pid':PID,'new_pid':new.pid,'verified_static_files':len(verified),'verified_bytes':sum(x['bytes'] for x in verified),'all_http_byte_equal':True,'endpoint_get_passed':True,'rollback_argv':OLD,'files':verified}
        (HERE/'feedback_live_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        (HERE/'feedback_server.pid').write_text(str(new.pid)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    except BaseException:
        new.terminate();new.wait(timeout=5)
        rollback=launch(OLD,'rollback_static_server.log');ready(rollback)
        raise
