#!/usr/bin/env python3
"""Independently verify saved efficiency calls; stdlib only, no inference."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,json,math,hashlib,statistics,struct,gzip

def read_csv(p):
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt',newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
    assert r and len({(x['episode'],x['frame_id']) for x in r})==len(r)
    return sorted(r,key=lambda x:(int(x['episode']),int(x['frame_id'])))
def stats(v):
    v=sorted(v)
    def q(p):
        k=(len(v)-1)*p;a=int(k);b=math.ceil(k);return v[a]+(v[b]-v[a])*(k-a)
    return {'n':len(v),'mean_ms':statistics.mean(v),'median_ms':statistics.median(v),'p95_ms':q(.95),'p99_ms':q(.99),'max_ms':max(v),'fraction_under_50ms':sum(x<=50 for x in v)/len(v)}
def file_for(root,arm,task,rnd):
    p=root/arm/f'{task}_v10_cpu_r{rnd}.csv'
    return p if p.exists() else p.with_suffix('.csv.gz')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    meta=json.loads(a.metadata.read_text())['run_metadata'];out={'scope':'R2 replay actions/phase/finished/batch equality; full state and closed-loop validation pending','tasks':{},'raw_sha256':{},'metadata_checks':{}};hosts=set();versions=set();affinities=set();tapes={}
    for task,n in [('2x5',316),('2x10',671)]:
        pv=[];cv=[];rounds=[];stages={'parent':defaultdict(list),'candidate':defaultdict(list)};phase_counts={};missing=set();batch_values=Counter();cold=[]
        for rnd in (1,2,3):
            p=file_for(a.raw,'parent',task,rnd);c=file_for(a.raw,'cand',task,rnd);pr,cr=read_csv(p),read_csv(c);assert len(pr)==len(cr)==n
            for arm,path,rows in [('parent',p,pr),('cand',c,cr)]:
                out['raw_sha256'][str(path.relative_to(a.raw))]=hashlib.sha256(path.read_bytes()).hexdigest()
                m=meta[f'r2_pair/{arm}/{task}_v10_cpu_r{rnd}.json'];assert m['calls']==n and m['tape_frames']==n and m['threads']==4
                assert m['simulation_steps']==m['render_calls']==0
                assert all(stop['frame_id']==n-1 and stop['call']==n-1 for stop in m['stopped_episodes'])
                hosts.add(m['host']);versions.add((m['python'],m['torch']));affinities.add(tuple(m['affinity']));tapes.setdefault(task,set()).add(m['tape_sha256'])
                assert abs(statistics.mean(float(x['latency_ms']) for x in rows)-m['all_calls']['mean_ms'])<1e-9
            for x,y in zip(pr,cr):
                for k in ['task','round','episode','call','frame_id','reference_phase','policy_phase_before','policy_phase_after','finished_before','finished_after','actual_batch_index']:
                    assert x[k]==y[k],(task,rnd,x['frame_id'],k)
                assert x['finished_before'].lower()=='false'
                aa,bb=json.loads(x['action_json']),json.loads(y['action_json']);assert len(aa)==len(bb)==14
                assert all(math.isfinite(z) for z in aa+bb)
                assert struct.pack('<14d',*aa)==struct.pack('<14d',*bb),(task,rnd,x['frame_id'],'action_bytes')
                u,v=float(x['latency_ms']),float(y['latency_ms']);assert u>=0 and v>=0 and math.isfinite(u+v)
                pv.append(u);cv.append(v);stages['parent'][x['policy_phase_before']].append(u);stages['candidate'][y['policy_phase_before']].append(v)
                assert x['actual_batch_index'] not in ('','None','null'); int(x['actual_batch_index'])
                batch_values[x['actual_batch_index']]+=1
                for k in ['decision_state_sha256','observation_sha256','state_schema_version']:
                    if not x.get(k) or not y.get(k):missing.add(k)
            assert [int(x['frame_id']) for x in pr]==list(range(n))
            pm=statistics.mean(float(x['latency_ms']) for x in pr);cm=statistics.mean(float(x['latency_ms']) for x in cr)
            rounds.append({'round':rnd,'parent_mean_ms':pm,'candidate_mean_ms':cm,'latency_reduction_pct':100*(1-cm/pm)})
        before,after=stats(pv),stats(cv)
        out['tasks'][task]={'distinct_observations':n,'repeated_calls_per_version':len(pv),'parent':before,'candidate':after,
            'latency_reduction_pct':100*(1-after['mean_ms']/before['mean_ms']), 'speedup_factor':before['mean_ms']/after['mean_ms'],
            'rounds':rounds,'actions_float64_bytes_equal':True,'phases_and_finished_equal':True,'missing_full_state_fields':sorted(missing),
            'actual_batch_index_values':dict(batch_values),'stage_costs':{arm:{name:{'calls':len(v),'total_ms':sum(v),'mean_ms':statistics.mean(v),'share':sum(v)/sum(pv if arm=='parent' else cv)} for name,v in ss.items()} for arm,ss in stages.items()}}
    assert len(hosts)==1 and len(affinities)==1 and all(len(x)==1 for x in tapes.values())
    out['metadata_checks']={'same_host':True,'same_affinity':True,'host':next(iter(hosts)),'affinity':next(iter(affinities)), 'tapes':{k:next(iter(v)) for k,v in tapes.items()},'versions':[list(v) for v in versions]}
    out['status']='VALID_REPLAY_SPEED_GAIN; CLOSED_LOOP_QUALITY_NOT_TESTED; REPORTED_LRU_NOT_IMPLEMENTED'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for t,d in out['tasks'].items():print(t,d['parent']['mean_ms'],d['candidate']['mean_ms'],d['latency_reduction_pct'],'missing',d['missing_full_state_fields'],'batches',d['actual_batch_index_values'])
if __name__=='__main__':main()
