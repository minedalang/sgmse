#!/usr/bin/env python3
"""Offline, standard-library-only assembly of fixed research drafts; no scientific execution."""
from pathlib import Path
import hashlib, html, json, re
from urllib.parse import urlsplit, unquote
from generate_figures import main as generate_figure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = [
 ('导读：研究问题与阶段性结论', 'THESIS_FRAMING_DRAFT_20261019.md', None),
 ('摘要：中文与English Abstract', 'THESIS_ABSTRACT_DRAFT_20261020.md', None),
 ('第一部分：绪论', 'THESIS_INTRODUCTION_DRAFT_20261020.md', None),
 ('第二部分：相关工作与研究定位', 'P8_RELATED_WORK_DRAFT_20261019.md', '## 6. 本轮版本补证与停止边界'),
 ('第三部分：已实现方法', 'METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md', None),
 ('第四部分：已完成实验结果', 'THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md', '## 本次写作来源校验'),
 ('第五部分：讨论与证据边界', 'P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md', '## 3. 本次写作核对范围'),
 ('第六部分：结论与展望', 'THESIS_CONCLUSION_DRAFT_20261020.md', None),
]
BIB = 'P8_RELATED_WORK_REFERENCES_20261019.bib'
LINK = re.compile(r'\[([^\]\n]+)\]\(([^\s)]+)\)')

def sha(b): return hashlib.sha256(b).hexdigest()
def relocate(md):
    def fix(m):
        url = m[2]
        if not urlsplit(url).scheme and not url.startswith(('/', '#')):
            url = '../' + url
        return '[' + m[1] + '](' + url + ')'
    return LINK.sub(fix, md)

def inline(s):
    # Protect code and links before escaping/emphasis; no raw source HTML is executed.
    tokens = []
    def hold(value):
        tokens.append(value)
        return '\x00' + str(len(tokens)-1) + '\x00'
    s = re.sub(r'`([^`]+)`', lambda m: hold('<code>'+html.escape(m[1])+'</code>'), s)
    def link(m):
        target = m[2]
        if urlsplit(target).scheme.lower() not in ('', 'http', 'https', 'file'):
            return hold(html.escape(m[0]))
        return hold('<a href="'+html.escape(target, quote=True)+'">'+html.escape(m[1])+'</a>')
    s = LINK.sub(link, s)
    s = html.escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    return re.sub('\x00([0-9]+)\x00', lambda m: tokens[int(m[1])], s)

def render(md):
    lines, out, toc = md.splitlines(), [], []
    i, heading_number = 0, 0
    while i < len(lines):
        line = lines[i]
        if not line.strip(): i += 1; continue
        if line.startswith('```') or line.strip() == r'\[':
            math = line.strip() == r'\['
            end = r'\]' if math else '```'
            block = [line] if math else []
            i += 1
            while i < len(lines) and not (lines[i].strip() == end if math else lines[i].startswith(end)):
                block.append(lines[i]); i += 1
            if i < len(lines):
                if math: block.append(lines[i])
                i += 1
            out.append('<pre class="'+('math-source' if math else 'code-block')+'">'+html.escape('\n'.join(block))+'</pre>')
            continue
        m = re.match(r'^(#{1,6}) (.*)', line)
        if m:
            level, title = len(m[1]), m[2]
            heading_number += 1
            anchor = 'section-' + str(heading_number)
            out.append(f'<h{level} id="{anchor}">'+inline(title)+f'</h{level}>')
            if level <= 3: toc.append(f'<li class="level-{level}"><a href="#{anchor}">'+html.escape(title)+'</a></li>')
            i += 1; continue
        if line.startswith('|') and i+1 < len(lines) and re.match(r'^\|[\s:|\-]+\|$', lines[i+1]):
            rows = [line]; i += 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]); i += 1
            table = '<div class="table-scroll"><table>'
            for n, row in enumerate(rows):
                tag = 'th' if n == 0 else 'td'
                table += '<tr>'+''.join('<'+tag+'>'+inline(cell.strip())+'</'+tag+'>' for cell in row.strip('|').split('|'))+'</tr>'
            out.append(table+'</table></div>'); continue
        if line.startswith('!['):
            image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if image_match and image_match[2] == 'figures/router_condition_tradeoffs.svg':
                out.append('<figure><img src="'+image_match[2]+'" alt="'+html.escape(image_match[1], quote=True)+'" style="width:100%;height:auto"><figcaption>'+html.escape(image_match[1])+'</figcaption></figure>')
                i += 1; continue
        if line.startswith('> '):
            out.append('<blockquote>'+inline(line[2:])+'</blockquote>'); i += 1; continue
        if re.match(r'^[-*] ', line):
            out.append('<ul>')
            while i < len(lines) and re.match(r'^[-*] ', lines[i]):
                out.append('<li>'+inline(lines[i][2:])+'</li>'); i += 1
            out.append('</ul>'); continue
        if re.match(r'^\d+\. ', line):
            out.append('<ol>')
            while i < len(lines) and re.match(r'^\d+\. ', lines[i]):
                out.append('<li>'+inline(re.sub(r'^\d+\. ', '', lines[i]))+'</li>'); i += 1
            out.append('</ol>'); continue
        out.append('<p>'+inline(line)+'</p>'); i += 1
    return '\n'.join(out), '\n'.join(toc)

CSS = '''body{font:17px/1.85 system-ui,"Noto Sans CJK SC",sans-serif;color:#18212b;background:#f4f6f8;margin:0}main{max-width:1000px;margin:auto;padding:2rem;background:white}nav{border:1px solid #b7c5d2;padding:1rem;background:#f5f9fd}nav ul{list-style:none;padding:0}.level-3{margin-left:1rem}h1,h2,h3,h4{line-height:1.4;scroll-margin-top:1rem}h2{border-bottom:2px solid #c9d8e6;padding-top:1rem}a{color:#145b9b;overflow-wrap:anywhere}blockquote{border-left:5px solid #d59b1f;padding:.5rem 1rem;background:#fff8e6;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:1rem;font:14px/1.65 ui-monospace,monospace}.math-source{border-left:4px solid #536aa1}code{background:#eef2f6;font-size:.9em}.table-scroll{overflow-x:auto}table{border-collapse:collapse;font-size:.9em;min-width:450px;width:100%}th,td{border:1px solid #c4cdd6;padding:.4rem;text-align:left;vertical-align:top}th{background:#eef3f8}@media(max-width:600px){main{padding:1rem}body{font-size:16px}}@media print{body{background:white;font-size:10pt}main{max-width:none;padding:0}nav{break-after:page}h2{break-before:page}h3,h4{break-after:avoid}pre,blockquote{background:white}.table-scroll{overflow:visible}table{min-width:0}a{color:inherit;text-decoration:none}}'''

def main():
    generate_figure()
    evidence_name = 'experiments/final_eval_full_20261018/metrics.json'
    evidence_before = (ROOT/evidence_name).read_bytes()
    names = [s[1] for s in SOURCES] + [BIB]
    before = {name: (ROOT/name).read_bytes() for name in names}
    texts = {name: b.decode('utf-8') for name,b in before.items()}
    chunks = ['# StoRM / SGMSE：已实现方法与已完成证据整合草稿\n\n'
      '> **不是完整毕业论文，也不是已验证新方法成功的宣告。** 本稿整合八份可编辑草稿，含中英文摘要、绪论与结论；学校模板、TeX编译及浏览器显示仍未验收，完整毕业论文所需其他部分尚未齐备。当前没有独立新机制收益确认，科研目标未完成；科学结论未因集成而扩张。\n\n'
      '> **公式显示限制**：本地未使用数学渲染器；公式保留原始 LaTeX 文本（灰底公式块），不声称数学排版或真实浏览器显示已验收。\n\n'
      '[阅读说明与复现](README.md) · [源文件及 SHA256 清单](sources_manifest.json)\n']
    appendices = []
    for title,name,marker in SOURCES:
        source = texts[name]
        body, sep, tail = source.partition(marker) if marker else (source,'','')
        # Only navigation and heading depth change; original scientific prose remains intact.
        def section(t):
            return re.sub(r'^(#{1,5}) ', r'#\1 ', relocate(t), flags=re.M)
        figure = ''
        if name == 'THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md':
            figure = '\n![图1：冻结原始逐条件区间，非speaker聚类/多重校正；不是新实验，DNSMOS非人耳MOS。](figures/router_condition_tradeoffs.svg)\n\n[独立打开SVG图](figures/router_condition_tradeoffs.svg) · [精确数据JSON](figures/router_condition_tradeoffs.data.json)。仅从冻结JSON转录七指标均值差及原95%CI，每metric独立轴，不含pooled、不排名或合成分数。\n\n'
        chunks.append('\n# '+title+'\n\n来源：['+name+'](../'+name+')\n\n'+figure+section(body))
        if sep: appendices.append('\n## '+name+'：原写作记录\n\n'+section(sep+tail))
    chunks.append('\n# 附录：原写作来源记录\n\n以下是原草稿的来源校验／版本说明，不是新增实验或正文结果。\n'+''.join(appendices))
    chunks.append('\n# 附录：固定版本参考文献\n\n[原始 BibTeX](../'+BIB+')；未作 BibTeX / TeX 编译。\n\n```bibtex\n'+texts[BIB]+'```\n')
    md = '\n'.join(chunks)
    body,toc = render(md)
    page = '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>StoRM / SGMSE 整合阅读稿（未完成论文）</title><style>'+CSS+'</style></head><body><main><nav aria-label="目录"><h2>目录</h2><ul>'+toc+'</ul></nav>'+body+'</main></body></html>\n'
    checks = {}
    checks['inline_code_escaped'] = inline('`<x>`') == '<code>&lt;x&gt;</code>'
    checks['relative_link_relocated'] = relocate('[x](experiments/a.md#b)') == '[x](../experiments/a.md#b)'
    checks['external_link_preserved'] = relocate('[x](https://example.org)') == '[x](https://example.org)'
    checks['table_rendered'] = '<table>' in render('| a | b |\n|---|---|\n| 1 | 2 |')[0]
    checks['math_verbatim_escaped'] = r'\[\nx < y\n\]'.replace(r'\n','\n') in html.unescape(render('\\[\nx < y\n\\]')[0])
    checks['no_scripts_or_external_assets'] = '<script' not in page and '<link' not in page and re.findall(r'<img src="([^"]+)"',page) == ['figures/router_condition_tradeoffs.svg']
    checks['frozen_figure_source_unchanged'] = (ROOT/evidence_name).read_bytes() == evidence_before
    checks['source_bytes_unchanged'] = all((ROOT/n).read_bytes()==b for n,b in before.items())
    checks['display_math_blocks_preserved'] = md.count('\\[') == sum(t.count('\\[') for n,t in texts.items() if n != BIB)
    missing=[]
    for m in LINK.finditer(md):
        u=urlsplit(m[2])
        if not u.scheme and u.path and not (HERE/unquote(u.path)).exists() and u.path not in ('README.md','sources_manifest.json'):
            missing.append(m[2])
    checks['local_link_targets_exist'] = not missing
    assert all(checks.values()), (checks,missing)
    (HERE/'manuscript.md').write_text(md,encoding='utf-8')
    (HERE/'manuscript.html').write_text(page,encoding='utf-8')
    manifest = {'kind':'documentation_integration_not_new_scientific_results','generator':'generate_reader.py','command':'cd /data/coding/sgmse && python3 -B thesis/generate_reader.py',
      'sources':[{'path':'../'+n,'bytes':len(b),'sha256':sha(b)} for n,b in before.items()]+[{'path':'../'+evidence_name,'bytes':len(evidence_before),'sha256':sha(evidence_before),'use':'Only original seven per-condition mean_delta/ci95 fields for static figure'}],
      'outputs':{n:{'bytes':(HERE/n).stat().st_size,'sha256':sha((HERE/n).read_bytes())} for n in ('manuscript.md','manuscript.html','figures/router_condition_tradeoffs.svg','figures/router_condition_tradeoffs.data.json')},
      'checks':checks,'figure_static_checks':json.loads((HERE/'figures/router_condition_tradeoffs.data.json').read_text())['static_checks'],'figure_generator_sha256':sha((HERE/'generate_figures.py').read_bytes()),'missing_local_links':missing,
      'unverified':['Real browser rendering, mobile and print layout','Mathematical typesetting; raw LaTeX is preserved, not rendered','BibTeX/TeX compilation and school thesis formatting','No revalidation of historical scientific results or external references'],
      'transformations':['Framing guide followed by fixed chapter ordering','Heading depth adjusted','Relative link targets prefixed ../ from source-root directory','Existing writing audit/version sections moved intact to appendix','BibTeX embedded verbatim as appendix'],
      'attachment_preflight':{'bytes':4694,'sha256':'75e785ebb749a77f37c91741fd2a0c7fc606416903ac9e6b1116d07dce12a613','note':'Read in full before implementation; historical uppercase research entry is superseded by current AGENTS routing'},
      'generator_sha256':sha(Path(__file__).read_bytes()),
      'dependencies':'Python standard library only; no Markdown packages installed or fetched'}
    (HERE/'sources_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    assert size <= 1024*1024, size
    print(json.dumps({'checks':checks,'total_bytes':size,'source_sha256':{n:sha(b) for n,b in before.items()}},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
