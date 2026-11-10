#!/usr/bin/env python3
"""Run one historical A1 regression grid with CPU physics and software rendering.

This uses native random first-grasp sampling and the unchanged public evaluator.
It never overwrites or removes previous outputs. No simulator is imported by --help.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = '0729068ee1e75a562cfbc4f2c83ab9539cf75081'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=('grid24000','grid620xxx'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    sim = Path(os.environ['SIM_REPO']).resolve()
    commit = subprocess.check_output(['git','-C',str(sim),'rev-parse','HEAD'],text=True).strip()
    if commit != PIN: raise SystemExit(f'Simulator commit mismatch: {commit}')
    if subprocess.check_output(['git','-C',str(sim),'status','--porcelain'],text=True).strip():
        raise SystemExit('Use a clean pinned simulator checkout')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_a1.py')],check=True)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    seeds = json.loads((ROOT/f'results/a1_regression_20261110/{args.grid}_seeds.json').read_text())['seeds']
    env = os.environ.copy()
    env.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',CUDA_VISIBLE_DEVICES='',LP_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1')
    env['PYTHONPATH'] = str(sim/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    jobs = [(t,p,s) for t in ('2x5','2x10') for p in ('basic','medium','advanced') for s in seeds]
    def run(job):
        task,profile,seed=job; label=f'{task}_{profile}_{seed}'
        command=[sys.executable,str(ROOT/'evaluation/evaluate_policy.py'),'--policy','policy:make_policy_b','--policy-code',str(ROOT/'candidates/a1'),'--output',str(output/label),'--tasks',task,'--profiles',profile,'--episodes','1','--seed-start',str(seed),'--max-steps','500','--max-steps-2x10','3200','--source-column','1','--success-hold-steps','5','--render-cameras']
        with (output/(label+'.log')).open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        return {'label':label,'returncode':result.returncode,'command':command}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run,jobs))
    (output/'execution_receipt.json').write_text(json.dumps({'scope':'previously seen development grid; not holdout','grid':args.grid,'simulator_commit':commit,'jobs':results},indent=2)+'\n')
    if any(r['returncode'] for r in results): raise SystemExit('One or more episodes failed to execute; inspect original logs')
    print(f'Finished {len(results)} executions; inspect native episodes.jsonl files for scores, not process exits.')

if __name__ == '__main__': main()
