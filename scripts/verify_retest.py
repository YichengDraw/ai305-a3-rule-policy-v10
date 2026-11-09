#!/usr/bin/env python3
"""Recalculate the second V10 seed-sample score using saved native outputs."""
from pathlib import Path
import csv,gzip,hashlib,json
R=Path(__file__).resolve().parents[1]
p=R/'results/retest'
manifest=json.loads((p/'manifest.json').read_text())
raw=gzip.decompress((p/'native_episodes.jsonl.gz').read_bytes())
assert hashlib.sha256(raw).hexdigest()==manifest['original_native_jsonl_sha256']
assert hashlib.sha256((p/'seeds.json').read_bytes()).hexdigest()==manifest['seed_spec_sha256']
records=[json.loads(line) for line in raw.splitlines() if line.strip()]
assert len(records)==manifest['native_episode_lines']==120
seeds=json.loads((p/'seeds.json').read_text())['seeds']
assert len(seeds)==20 and len(set(seeds))==20
with (p/'episodes.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==120
assert {(x['task'],x['profile'],int(x['seed'])) for x in rows}=={(t,v,s) for t in ('2x5','2x10') for v in ('basic','medium','advanced') for s in seeds}
original={(x['task'],x['profile'],int(x['seed'])):x for x in records}
assert len(original)==120
completion=0;success=0
for t in ('2x5','2x10'):
 for v in ('basic','medium','advanced'):
  z=[x for x in rows if x['task']==t and x['profile']==v]
  count=sum(int(x['cookies_in_target']) for x in z)
  cap=sum(int(x['capacity']) for x in z)
  wins=sum(x['official_success'].lower()=='true' for x in z)
  assert len(z)==20 and cap==(200 if t=='2x5' else 400)
  for x in z:
   old=original[(t,v,int(x['seed']))]
   assert int(x['cookies_in_target'])==old['cookies_in_target']
   assert (x['official_success'].lower()=='true')==old['success']
  completion+=count/cap;success+=wins/20
  print(f'{t:4s} {v:8s} {count}/{cap} {count/cap*100:.2f}% {wins}/20')
completion/=6;success/=6
assert abs(completion-0.96625)<1e-12,completion
assert abs(success-113/120)<1e-12,success
print(f'PASS: 120 native episodes; equal-setting completion={completion*100:.3f}% success={success*100:.3f}%')
