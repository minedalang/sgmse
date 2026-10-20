#!/usr/bin/env python3
"""One static figure from frozen summary fields; no estimation or scientific execution."""
from pathlib import Path
import hashlib, json, math, html
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'experiments/final_eval_full_20261018/metrics.json'
EXPECTED='15cf7076a0e233042e83e68ff429ff0b949e83ebcaf88297d1448d10b363daaa'
METRICS=['PESQ_e','STOI_e','SI-SDR_e','DNSMOS_SIG_e','DNSMOS_OVRL_e','DNSMOS_P808_e','DNSMOS_BAK_e']
SPLITS=['no_reverb','with_reverb']
LABELS={'no_reverb':'无混响 / no-reverb','with_reverb':'有混响 / with-reverb'}
COLORS={'no_reverb':'#14659e','with_reverb':'#b14d13'}
def digest(b):return hashlib.sha256(b).hexdigest()
def number(x):return format(x,'.17g')
def main():
 b=SOURCE.read_bytes();assert digest(b)==EXPECTED
 d=json.loads(b)['paired_bootstrap']['splits']
 selected={s:{m:{'mean_delta':d[s]['router_vs_off'][m]['mean_delta'],'ci95':d[s]['router_vs_off'][m]['ci95']} for m in METRICS} for s in SPLITS}
 for s in SPLITS:
  for m in METRICS:
   v=selected[s][m];assert all(math.isfinite(x) for x in [v['mean_delta']]+v['ci95']);assert v['ci95'][0]<=v['mean_delta']<=v['ci95'][1]
 record={'source_path':'../../experiments/final_eval_full_20261018/metrics.json','source_sha256':EXPECTED,'field_path':'paired_bootstrap.splits.{no_reverb,with_reverb}.router_vs_off','contrast':'Router minus off','paired_cases_per_condition':150,'metric_order':METRICS,'data':selected,'note':'冻结原始逐条件区间，非speaker聚类/多重校正；不是新实验，DNSMOS非人耳MOS。每个metric独立轴；不含pooled，不排名或合成分数。JSON保留源浮点值，图中数字17位有效数字。'}
 out=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1510" viewBox="0 0 1200 1510" role="img" aria-labelledby="title desc">','<title id="title">Router条件性权衡：七指标的逐条件均值差与原95%区间</title>','<desc id="desc">两条件各150个配对，Router减off。每指标独立横轴，蓝色无混响、棕色有混响。点为均值，线段为冻结原始95%置信区间，虚线为零。所有精确数值见下方文字和配套JSON。非speaker聚类、无多重比较校正、不是新实验，DNSMOS不是人耳MOS。</desc>','<rect width="1200" height="1510" fill="white"/>','<g font-family="system-ui, sans-serif" fill="#18212b">']
 def text(x,y,t,size=16,fill=None):out.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(f' fill="{fill}"' if fill else '')+'>'+html.escape(t)+'</text>')
 def line(x1,y1,x2,y2,color='#555',width=1,dash=False):out.append(f'<line x1="{x1:.5f}" y1="{y1:.5f}" x2="{x2:.5f}" y2="{y2:.5f}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
 text(30,35,'Router − off：逐条件重建与质量权衡（7个独立指标轴）',26)
 text(30,65,'两条件各150配对；点＝均值差，线段＝冻结原95%CI，虚线＝0。不是新实验。',17)
 text(30,91,'冻结原始逐条件区间，非speaker聚类/多重校正；DNSMOS非人耳MOS。',17)
 text(30,115,'横轴单位：SI-SDR为dB，其余为各自指标分值；正/负仅表示Router相对off的差。',15)
 for i,m in enumerate(METRICS):
  y=135+i*190
  vals=[0.0]+[x for s in SPLITS for x in selected[s][m]['ci95']]
  lo,hi=min(vals),max(vals);pad=(hi-lo)*.12;lo-=pad;hi+=pad
  def x(v):return 240+(v-lo)/(hi-lo)*480
  name=m.removesuffix('_e').replace('_','-');unit='dB' if m=='SI-SDR_e' else '指标分值'
  text(30,y+23,name+'  /  '+unit,21)
  line(x(0),y+32,x(0),y+102,'#777',1,True)
  for j,s in enumerate(SPLITS):
   v=selected[s][m]; yy=y+48+j*35;left,right=v['ci95'];color=COLORS[s]
   text(30,yy+5,LABELS[s],16,color)
   line(x(left),yy,x(right),yy,color,3);line(x(left),yy-6,x(left),yy+6,color,2);line(x(right),yy-6,x(right),yy+6,color,2)
   out.append(f'<circle cx="{x(v["mean_delta"]):.5f}" cy="{yy}" r="5" fill="{color}"/>')
   text(750,yy+5,'均值 '+number(v['mean_delta']),16,color)
   text(30,y+147+j*22,LABELS[s]+' 原CI ['+number(left)+', '+number(right)+']',15,color)
  line(240,y+110,720,y+110,'#444')
  for v in (lo,0.0,hi):
   xx=x(v);line(xx,y+106,xx,y+114,'#444');text(xx-25,y+131,format(v,'.5g'),13)
  line(30,y+183,1170,y+183,'#d9dfe5')
 text(30,1490,'数据：final_eval_full_20261018/metrics.json；未重采样、未重算指标；同图data.json保留精确源数值。',14)
 out+=['</g>','</svg>'];svg='\n'.join(out)+'\n'
 tree=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'}
 checks={'static_xml_parses':True,'fourteen_mean_points':len(tree.findall('.//s:circle',ns))==14,'title_and_desc':all(tree.find('s:'+n,ns) is not None for n in ('title','desc')),'no_external_references':not any(k.endswith('href') or k=='src' for e in tree.iter() for k in e.attrib),'no_script':not tree.findall('.//s:script',ns),'source_bytes_unchanged':SOURCE.read_bytes()==b,'all_14_exact_value_records':all(selected[s][m]['mean_delta']==d[s]['router_vs_off'][m]['mean_delta'] and selected[s][m]['ci95']==d[s]['router_vs_off'][m]['ci95'] for s in SPLITS for m in METRICS)}
 assert all(checks.values()),checks
 record['static_checks']=checks
 folder=HERE/'figures';folder.mkdir(exist_ok=True)
 (folder/'router_condition_tradeoffs.svg').write_text(svg,encoding='utf-8')
 (folder/'router_condition_tradeoffs.data.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
