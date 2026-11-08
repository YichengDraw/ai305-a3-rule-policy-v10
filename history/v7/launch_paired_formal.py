"""Run a frozen candidate and the frozen V6 reference on the registered paired test set."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--new-code',type=Path,required=True)
    p.add_argument('--old-code',type=Path,required=True)
    p.add_argument('--new-policy',default='policy:make_policy_b')
    p.add_argument('--old-policy',default='policy:make_policy_b')
    p.add_argument('--old-label',default='v6')
    p.add_argument('--new-label',default='v7')
    p.add_argument('--seed-start',type=int,default=16000)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=36)
    p.add_argument('--shard-size',type=int,default=1)
    p.add_argument('--egl-devices',nargs='+',type=int,default=[2,4])
    args=p.parse_args()
    if args.workers<2:p.error('At least two total workers are required')
    if args.old_label==args.new_label:p.error('Old and new labels must differ')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    jobs=[dict(name=args.old_label,code=str(args.old_code.resolve()),policy=args.old_policy,workers=args.workers//2),
          dict(name=args.new_label,code=str(args.new_code.resolve()),policy=args.new_policy,workers=args.workers-args.workers//2)]
    protocol=dict(scope='frozen_paired_formal_test',started_utc=datetime.now(timezone.utc).isoformat(),
        seed_start=args.seed_start,episodes_per_cell=args.episodes,planned_episodes_per_policy=9*args.episodes,
        steps={'2x5':500,'2x10':3200},source_column=1,success_hold_steps=5,
        strict_policy_completion=False,shard_size=args.shard_size,
        selection_rule='candidate selected on development data before this test',
        tables={'extended':dict(episodes_per_policy=6*args.episodes,steps={'2x5':500,'2x10':3200}),
                'official500':dict(episodes_per_policy=6*args.episodes,steps={'2x5':500,'2x10':500},
                    reused_2x5_from_extended=3*args.episodes,additional_2x10=3*args.episodes)},jobs=jobs)
    (root/'paired_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.monotonic()

    def run(job):
        command=[sys.executable,str(scripts/'launch_evaluation.py'),'--output',str(root/job['name']),
            '--policy-code',job['code'],'--policy',job['policy'],'--workers',str(job['workers']),
            '--seed-start',str(args.seed_start),'--episodes',str(args.episodes),'--shard-size',str(args.shard_size),
            '--extra-2x10-500','--egl-devices',*[str(g) for g in args.egl_devices]]
        with (root/(job['name']+'.log')).open('w') as log:
            proc=subprocess.run(command,cwd=scripts,stdout=log,stderr=subprocess.STDOUT)
        return dict(job,returncode=proc.returncode,command=command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,jobs))
    (root/'paired_status.json').write_text(json.dumps(dict(results=results,wall_seconds=time.monotonic()-start),indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit('Formal subprocess failure: inspect saved per-policy logs')
    for job in jobs:
        policy_root=root/job['name']
        inputs=sorted((policy_root/'default').glob('2x5_*'))+sorted((policy_root/'extra_2x10_500').glob('2x10_*'))
        inputs=[p for p in inputs if p.is_dir()]
        if len(inputs)!=6*args.episodes:raise RuntimeError('Missing official500 shard inputs')
        subprocess.run([sys.executable,str(scripts/'merge_results.py'),'--inputs',*[str(p) for p in inputs],
            '--output',str(policy_root/'final_official500'),'--episodes',str(args.episodes),
            '--seed-start',str(args.seed_start),'--max-steps','500','--max-steps-2x10','500'],cwd=scripts,check=True)
    subprocess.run([sys.executable,str(scripts/'paired_comparison.py'),
        '--old',str(root/args.old_label/'final_default'),'--new',str(root/args.new_label/'final_default'),
        '--old-label',args.old_label.upper(),'--new-label',args.new_label.upper(),
        '--seed-start',str(args.seed_start),'--episodes',str(args.episodes),
        '--protocol-label','方案扩展500/3200','--output',str(root/'comparison')],cwd=scripts,check=True)
    subprocess.run([sys.executable,str(scripts/'paired_comparison.py'),
        '--old',str(root/args.old_label/'final_official500'),'--new',str(root/args.new_label/'final_official500'),
        '--old-label',args.old_label.upper(),'--new-label',args.new_label.upper(),
        '--seed-start',str(args.seed_start),'--episodes',str(args.episodes),'--max-steps-2x10','500',
        '--protocol-label','教师统一500步','--output',str(root/'comparison_official500')],cwd=scripts,check=True)
    (root/'complete.json').write_text(json.dumps(dict(status='completed',episodes=18*args.episodes,table_rows_per_policy=6*args.episodes,reused_2x5_per_policy=3*args.episodes,
        wall_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':main()
