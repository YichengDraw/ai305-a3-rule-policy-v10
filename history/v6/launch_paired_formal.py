"""Run a frozen candidate and the original v5 on the registered paired test set."""
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
    p.add_argument('--new-policy',default='policy:make_policy_c')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=36)
    p.add_argument('--shard-size',type=int,default=1)
    p.add_argument('--egl-devices',nargs='+',type=int,default=[2,4])
    args=p.parse_args()
    if args.workers<2:p.error('At least two total workers are required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    jobs=[dict(name='v5',code=str(args.old_code.resolve()),policy='policy:make_policy',workers=args.workers//2),
          dict(name='v6',code=str(args.new_code.resolve()),policy=args.new_policy,workers=args.workers-args.workers//2)]
    protocol=dict(scope='frozen_paired_formal_test',started_utc=datetime.now(timezone.utc).isoformat(),
        seed_start=15000,episodes_per_cell=20,planned_episodes_per_policy=120,
        steps={'2x5':500,'2x10':3200},source_column=1,success_hold_steps=5,
        strict_policy_completion=False,shard_size=args.shard_size,
        selection_rule='candidate selected on development data before this test',jobs=jobs)
    (root/'paired_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.monotonic()

    def run(job):
        command=[sys.executable,str(scripts/'launch_evaluation.py'),'--output',str(root/job['name']),
            '--policy-code',job['code'],'--policy',job['policy'],'--workers',str(job['workers']),
            '--seed-start','15000','--episodes','20','--shard-size',str(args.shard_size),
            '--default-budgets-only','--egl-devices',*[str(g) for g in args.egl_devices]]
        with (root/(job['name']+'.log')).open('w') as log:
            proc=subprocess.run(command,cwd=scripts,stdout=log,stderr=subprocess.STDOUT)
        return dict(job,returncode=proc.returncode,command=command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,jobs))
    (root/'paired_status.json').write_text(json.dumps(dict(results=results,wall_seconds=time.monotonic()-start),indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit('Formal subprocess failure: inspect saved per-policy logs')
    subprocess.run([sys.executable,str(scripts/'paired_comparison.py'),
        '--old',str(root/'v5/final_default'),'--new',str(root/'v6/final_default'),
        '--output',str(root/'comparison')],cwd=scripts,check=True)
    (root/'complete.json').write_text(json.dumps(dict(status='completed',episodes=240,
        wall_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':main()
