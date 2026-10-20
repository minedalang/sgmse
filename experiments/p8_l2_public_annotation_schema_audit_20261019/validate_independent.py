# Independent line-state implementation; does not import business parser or share its functions/regex.
from pathlib import Path
from decimal import Decimal
import json,hashlib
B=Path(__file__).resolve().parent; raw=(B/'raw_textgrid.txt').read_bytes(); tiers=[]; current=None; interval=None; declared_global=None; global_bounds={}
for line in raw.decode('utf-8').splitlines():
    s=line.strip()
    if s.startswith('item [') and s!='item []:':
        current={'index':int(s[len('item ['):].split(']')[0]),'intervals':[]}; tiers.append(current); interval=None
    elif s.startswith('intervals ['):
        interval={'index':int(s[len('intervals ['):].split(']')[0])}; current['intervals'].append(interval)
    elif ' = ' in s:
        key,value=s.split(' = ',1)
        if current is None:
            if key=='size': declared_global=int(value)
            elif key in ['xmin','xmax']: global_bounds[key]=value
        elif key=='intervals: size': current['declared_intervals']=int(value)
        elif key in ['class','name']: current[key]=value[1:-1].replace('""','"')
        elif key in ['xmin','xmax']: (interval if interval is not None else current)[key]=value
        elif key=='text': interval['raw_label']=value[1:-1].replace('""','"')
stored=json.loads((B/'structure.json').read_text()); assert hashlib.sha256(raw).hexdigest()==stored['source_sha256']; assert len(tiers)==declared_global==stored['declared_tiers']
assert tiers==[{k:v for k,v in t.items() if k!='empty_label_count'} for t in stored['tiers']]
for t in tiers:
    assert t['declared_intervals']==len(t['intervals']); assert t['class']=='IntervalTier'
    assert t['xmin']==global_bounds['xmin'] and t['xmax']==global_bounds['xmax']; last=Decimal(t['xmin'])
    for idx,i in enumerate(t['intervals'],1):
        a,z=Decimal(i['xmin']),Decimal(i['xmax']); assert i['index']==idx and a==last and 0<=a<=z<=Decimal(t['xmax']); last=z
    assert last==Decimal(t['xmax'])
locks=json.loads((B/'protected_hashes.json').read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in locks.items())
receipt={'status':'passed','parser_independence':'separate line-state parser, no shared parser imports/functions/regex; shared raw source and published structure schema, same author and standard library','tiers':len(tiers),'interval_counts':{t['name']:len(t['intervals']) for t in tiers},'all_intervals_compared':sum(len(t['intervals']) for t in tiers),'all_times_Decimal_checked':True,'all_raw_labels_compared':True,'protected_hash_count':len(locks),'source_sha256':hashlib.sha256(raw).hexdigest()}
(B/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt))
