from pathlib import Path
import re,json,hashlib
from decimal import Decimal
from collections import Counter
B=Path(__file__).resolve().parent
text=(B/'raw_textgrid.txt').read_bytes().decode('utf-8')
def number(block,key):
    return re.search(r'^\s*'+re.escape(key)+r' = ([0-9.]+)\s*$',block,re.M).group(1)
def label(block,key):
    return re.search(r'^\s*'+key+r' = "((?:[^"]|"")*)"\s*$',block,re.M).group(1).replace('""','"')
parts=re.split(r'^    item \[(\d+)\]:\s*$',text,flags=re.M)
header=parts[0]; result={'source_sha256':hashlib.sha256((B/'raw_textgrid.txt').read_bytes()).hexdigest(),'global_xmin':number(header,'xmin'),'global_xmax':number(header,'xmax'),'declared_tiers':int(number(header,'size')),'tiers':[],'checks':[]}
for k in range(1,len(parts),2):
    idx=int(parts[k]); block=parts[k+1]; ip=re.split(r'^        intervals \[(\d+)\]:\s*$',block,flags=re.M); meta=ip[0]
    tier={'index':idx,'class':label(meta,'class'),'name':label(meta,'name'),'xmin':number(meta,'xmin'),'xmax':number(meta,'xmax'),'declared_intervals':int(number(meta,'intervals: size')),'intervals':[]}
    for j in range(1,len(ip),2): tier['intervals'].append({'index':int(ip[j]),'xmin':number(ip[j+1],'xmin'),'xmax':number(ip[j+1],'xmax'),'raw_label':label(ip[j+1],'text')})
    ints=tier['intervals']; assert tier['class']=='IntervalTier'; assert len(ints)==tier['declared_intervals']; assert [i['index'] for i in ints]==list(range(1,len(ints)+1))
    assert Decimal(tier['xmin'])==Decimal(result['global_xmin']) and Decimal(tier['xmax'])==Decimal(result['global_xmax'])
    assert Decimal(ints[0]['xmin'])==Decimal(tier['xmin']) and Decimal(ints[-1]['xmax'])==Decimal(tier['xmax'])
    for n,i in enumerate(ints):
        assert Decimal(0)<=Decimal(i['xmin'])<=Decimal(i['xmax'])<=Decimal(tier['xmax'])
        if n: assert Decimal(ints[n-1]['xmax'])==Decimal(i['xmin'])
    tier['empty_label_count']=sum(i['raw_label']=='' for i in ints); result['tiers'].append(tier)
assert len(result['tiers'])==result['declared_tiers']
phones=next(t for t in result['tiers'] if t['name']=='phones'); classes={k:[] for k in ['empty','no_comma','triple_s','triple_a','triple_d','unknown']}
for i in phones['intervals']:
    s=i['raw_label']; fields=s.split(',')
    c='empty' if s=='' else 'no_comma' if ',' not in s else 'triple_'+fields[2] if len(fields)==3 and fields[2] in ['s','a','d'] else 'unknown'
    classes[c].append(i)
result['phone_label_inventory']={k:{'count':len(v),'examples_all':v} for k,v in classes.items()}; result['checks']=['counts','indices','Decimal nonnegative ordered ranges','tier/global bounds','all adjacent continuity','all raw labels retained']
(B/'structure.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print([(t['name'],len(t['intervals']),t['empty_label_count']) for t in result['tiers']]); print({k:len(v) for k,v in classes.items()})
