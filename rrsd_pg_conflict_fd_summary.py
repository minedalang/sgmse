#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} LOG")
rx = re.compile(r"\[RRSD-FD\].*?t=(?P<t>[-+0-9.eE]+).*?conflict_tiles=(?P<n>\d+).*?plus_ok=(?P<po>[01]).*?minus_ok=(?P<mo>[01])")
rows = []
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        m = rx.search(line)
        if m:
            d = {k: float(v) for k, v in m.groupdict().items() if k == "t"}
            d.update({"n": int(m["n"]), "po": int(m["po"]), "mo": int(m["mo"])})
            rows.append(d)
print(f"Parsed targeted FD rows: {len(rows)}")
if not rows:
    raise SystemExit("No [RRSD-FD] rows found")
for name, pred in (("ALL", lambda t: True), ("LOW [0.05,0.2)", lambda t: .05 <= t < .2), ("MID [0.2,0.5)", lambda t: .2 <= t < .5)):
    x = [r for r in rows if pred(r["t"])]
    if not x:
        continue
    print(f"\n{name}: n={len(x)}")
    print(f"  rows with selected conflict tiles: {sum(r['n']>0 for r in x)}/{len(x)}")
    print(f"  mean selected tiles: {sum(r['n'] for r in x)/len(x):.3f}")
    print(f"  alpha increase: Future down + SSL up = {sum(r['po'] for r in x)}/{len(x)} ({sum(r['po'] for r in x)/len(x):.3f})")
    print(f"  alpha decrease: Future up + SSL down = {sum(r['mo'] for r in x)}/{len(x)} ({sum(r['mo'] for r in x)/len(x):.3f})")
    both = sum(r['po'] and r['mo'] for r in x)
    print(f"  bidirectional confirmation: {both}/{len(x)} ({both/len(x):.3f})")
