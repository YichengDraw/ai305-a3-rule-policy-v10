#!/usr/bin/env python3
"""Verify saved protection cases and exploratory timings; no policy or simulator."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics,struct
R=Path(__file__).resolve().parents[1]
m=json.loads((R/'manifest.json').read_text());s=json.loads((R/'reviewed_summary.json').read_text())
for f,h in m['raw_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h
q=json.loads(gzip.decompress((R/'raw/native_quality_selected.json.gz').read_bytes()))['cases']
assert len(q)==8 and len({x['case'] for x in q})==8
success=0
for c in q:
    p,n=c['parent'],c['candidate'];pr=c['protocol']
    assert all(p[k]==n[k] for k in ['success','cookies_in_target','cookies_in_source','steps'])
    a,b=dict(p['randomization']),dict(n['randomization'])
    for d in (a,b):
        for k in ('cookies_in_target','cookies_in_source','target_column_counts'):d.pop(k,None)
    assert a==b
    assert p.get('task_instruction',a.get('task_instruction'))==n['task_instruction']
    assert pr['source_commit']=='0729068ee1e75a562cfbc4f2c83ab9539cf75081'
    assert pr['strict_policy_completion'] is False and pr['success_hold_steps']==5
    assert pr['max_steps_by_task']=={'2x5':500,'2x10':3200}
    assert pr['source_column']=='1' and pr['first_grasp_2x10']=='random'
    assert c['runtime_sha256']==m['runtime_sha256']
    success+=int(n['success'])
assert success==4
for x in s['exploratory_probes']:
    folder='P21_test' if x['candidate']=='P2_1_search_context' else 'P22_test'
    filename=x['task']+'_v10_cpu_r1.csv.gz'
    p=R/'raw'/(folder+'_'+filename)
    with gzip.open(p,'rt',newline='') as f:a=list(csv.DictReader(f))
    with gzip.open(R.parent/'efficiency_round2/raw/cand'/filename,'rt',newline='') as f:b=list(csv.DictReader(f))
    assert len(a)==len(b)==x['n']
    for u,v in zip(a,b):
        for k in ('frame_id','policy_phase_before','policy_phase_after','actual_batch_index','finished_before','finished_after'):assert u[k]==v[k]
        aa,bb=json.loads(u['action_json']),json.loads(v['action_json'])
        assert len(aa)==len(bb)==14 and all(math.isfinite(z) for z in aa+bb)
        assert struct.pack('<14d',*aa)==struct.pack('<14d',*bb)
    assert abs(statistics.mean(float(z['latency_ms']) for z in a)-x['mean_ms'])<1e-8
print('PASS: eight quality outcomes and initializations match; four original successes retained and four original failures unchanged.')
print('PASS: four exploratory tapes match recorded actions/state fields; single-pass timings recomputed. No new accepted speedup.')
