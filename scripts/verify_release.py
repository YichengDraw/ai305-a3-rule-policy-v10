#!/usr/bin/env python3
"""Offline verification of frozen policy hashes and exported official terminal metrics."""
import ast, csv, hashlib, json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
policy=root/'frozen_policy_v10'
node=ast.parse((policy/'policy.py').read_text())
deps=next(ast.literal_eval(x.value) for x in node.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_DEPENDENCIES' for t in x.targets))
manifest=json.loads((policy/'freeze_manifest.json').read_text())
# Independently frozen hash dictionary is also embedded in the original freeze manifest.
expected=manifest.get('runtime_sha256') or manifest.get('sha256')
if not isinstance(expected,dict):
    raise RuntimeError('Missing runtime hash map in original frozen manifest')
assert len(deps)==78 and set(deps)==set(expected), (len(deps),len(expected))
for path in deps:
    actual=hashlib.sha256((policy/path).read_bytes()).hexdigest()
    assert actual==expected[path], f'Hash mismatch: {path}'
with (root/'results/v10_holdout_episodes.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==120
assert len({(r['task'],r['profile'],r['seed']) for r in rows})==120
assert {int(r['seed']) for r in rows}==set(range(24000,24020))
assert all(int(r['source_column'])==1 for r in rows)
assert all(r['official_success'] in ('true','false') for r in rows)
summary=json.loads((root/'results/v10_official_grid_summary.json').read_text())
cells=[]
for task in ('2x5','2x10'):
 for profile in ('basic','medium','advanced'):
  v=[r for r in rows if r['task']==task and r['profile']==profile]
  assert len(v)==20
  total=sum(int(r['cookies_in_target']) for r in v);cap=sum(int(r['capacity']) for r in v)
  success=sum(r['official_success']=='true' for r in v)
  a=next(c for c in summary['cells'] if c['task']==task and c['profile']==profile)
  assert (total,cap,success)==(a['cookies_in_target'],a['cookie_capacity'],a['successes'])
  cells.append((total/cap,success/len(v)))
completion=sum(x[0] for x in cells)/6
success=sum(x[1] for x in cells)/6
assert abs(completion-summary['overall']['equal_cell_mean_completion_rate'])<1e-12
assert abs(success-summary['overall']['equal_cell_mean_official_success_rate'])<1e-12
print(f'PASS: {len(deps)} frozen runtime SHA256 matches; {len(rows)} real V10 episodes; completion={completion:.4%}; success={success:.4%}')
