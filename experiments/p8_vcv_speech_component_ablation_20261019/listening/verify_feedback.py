"""Real HTTP checks against temporary 8767; tagged test records are retained."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

HERE = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8767/'
API = BASE + HERE.parent.name + '/listening/api/feedback'
checks = []
def call(url, data=None, expected=200, headers=None):
    request = Request(url, data=None if data is None else json.dumps(data).encode(), headers=headers or {'Content-Type': 'application/json'})
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as e:
        response = e
    assert response.code == expected, (response.code, expected, response.read())
    return json.loads(response.read())

if __name__ == '__main__':
    proc = subprocess.Popen([sys.executable, str(HERE/'feedback_server.py'), '--port', '8767', '--test-mode'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if proc.poll() is not None:
                raise RuntimeError('temporary server failed to start')
            try:
                urlopen(BASE, timeout=.2).close()
                break
            except OSError:
                time.sleep(.05)
        session_id = str(uuid.uuid4())
        ids = ['original__N1_snr-6__m2ada1','fresh__N1_snr-6__f20aka1','original__N2_snr-6__m2aga1']
        data = {'session_id':session_id,'response_id':str(uuid.uuid4()),'choices':dict(zip(ids,['similar','native','uncertain'])),'note':'AUTOMATED TEST ONLY; not listener evidence'}
        assert call(API+'?session_id='+session_id)['record'] is None
        checks.append('new session GET empty')
        first = call(API,data,201)['record']
        assert first['test_record'] and first['protocol']=='prompted/nonblind/singlelistener'
        checks.append('valid POST durable tagged record')
        assert call(API+'?session_id='+session_id)['record']==first
        checks.append('GET restores exact choices and IDs')
        retry = call(API,data)
        assert retry['duplicate'] and retry['record']==first
        checks.append('same-response retry idempotent')
        changed = copy.deepcopy(data);changed['note']='conflict'
        call(API,changed,409);checks.append('same ID different content rejected')
        changed['response_id']=str(uuid.uuid4())
        second=call(API,changed,201)['record']
        assert call(API+'?session_id='+session_id)['record']==second
        checks.append('latest by session; prior record remains')
        supplement = dict(data, response_id=str(uuid.uuid4()), choices={'original__N1_snr-6__f11ata1':'similar'})
        merged = call(API,supplement,201)['record']
        assert all(merged['choices'][k] == v for k,v in data['choices'].items())
        assert merged['choices']['original__N1_snr-6__f11ata1'] == 'similar'
        assert call(API,supplement)['duplicate']
        supplement2 = dict(data, response_id=str(uuid.uuid4()), choices={'fresh__N2_snr-6__f20aka1':'uncertain'})
        merged2 = call(API,supplement2,201)['record']
        assert len(merged2['choices']) == 5
        assert all(merged2['choices'][k] == v for k,v in merged['choices'].items())
        assert call(API+'?session_id='+session_id)['record'] == merged2
        checks.append('partial case4/5 saves merge prior three; retries do not erase choices')
        invalid=copy.deepcopy(data);invalid['session_id']='../../bad';call(API,invalid,400)
        invalid=copy.deepcopy(data);invalid['choices'][ids[0]]='fake';call(API,invalid,400)
        invalid=copy.deepcopy(data);invalid['choices']['unknown']='native';call(API,invalid,400)
        invalid=copy.deepcopy(data);invalid['note']='x'*2001;call(API,invalid,400)
        invalid=copy.deepcopy(data);invalid['note']='x'*17000;call(API,invalid,413)
        call(API+'?session_id=../../bad',expected=400)
        call(API,data,415,{'Content-Type':'text/plain'})
        call(API,data,403,{'Content-Type':'application/json','Origin':'http://untrusted.invalid'})
        checks.append('invalid IDs/values/cases/note/oversize/type/origin rejected')
        call(BASE+'arbitrary-write',data,404)
        call(BASE+'../outside',expected=403)
        call(BASE+HERE.parent.name+'/listening/responses/test_records.jsonl',expected=403)
        checks.append('arbitrary endpoints/traversal/record static reads rejected')
        for name in ['index.html','feedback.js',ids[0]+'__clean.wav']:
            assert urlopen(BASE+HERE.parent.name+'/listening/'+name).read()==(HERE/name).read_bytes()
        checks.append('static page/script/WAV byte equality')
        rows=[json.loads(line) for line in (HERE/'responses/test_records.jsonl').read_text().splitlines()]
        assert sum(r['session_id']==session_id for r in rows)==4
        checks.append('exactly four append-only records despite retries')
        result={'status':'passed','port':8767,'test_session_id':session_id,'checks':checks,'test_records':'responses/test_records.jsonl; all tagged test_record=true; never listener evidence'}
        (HERE/'feedback_test_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    finally:
        proc.terminate()
        proc.wait(timeout=5)
