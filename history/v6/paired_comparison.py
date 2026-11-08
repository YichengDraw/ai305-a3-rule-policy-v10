"""Recompute official scores from two matched episode files."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def load(path):
    if path.is_dir():
        path = path/'episodes.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed={(r['task'],r['profile'],r['seed']):r for r in rows}
    if len(indexed)!=len(rows):
        raise ValueError(f'Duplicate episode keys: {path}')
    return indexed


def metrics(rows):
    n=len(rows)
    return dict(episodes=n,successes=sum(bool(r['success']) for r in rows),
        success_rate=sum(bool(r['success']) for r in rows)/n,
        completion_rate=sum(r['cookies_in_target']/r['max_score'] for r in rows)/n,
        cookie_weighted_completion_rate=sum(r['cookies_in_target'] for r in rows)/sum(r['max_score'] for r in rows),
        target_total=sum(r['cookies_in_target'] for r in rows),
        mean_steps=sum(r['steps'] for r in rows)/n,
        done_but_failed=sum(r.get('phase')=='DONE' and not r['success'] for r in rows),
        phase_counts=dict(Counter(r.get('phase','unknown') for r in rows)),
        failure_reasons=dict(Counter(r.get('policy_failure_reason') or 'no controller failure reason'
            for r in rows if not r['success'])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=15000)
    p.add_argument('--episodes',type=int,default=20)
    args=p.parse_args()
    old,new=load(args.old),load(args.new)
    expected={(t,p,s) for t in ('2x5','2x10') for p in ('basic','medium','advanced')
        for s in range(args.seed_start,args.seed_start+args.episodes)}
    if set(old)!=expected or set(new)!=expected:
        raise ValueError('Episode keys do not match the preassigned 6-cell protocol')
    for key in expected:
        a,b=old[key],new[key]
        for field in ('max_steps','task_instruction','source_column','strict_policy_completion','success_hold_steps'):
            if a.get(field)!=b.get(field):
                raise ValueError(f'Paired protocol mismatch: {key} {field}')
        if a['max_steps']!=(500 if key[0]=='2x5' else 3200):
            raise ValueError(f'Unexpected step budget: {key}')
    ordered=sorted(expected)
    cells=[]
    for task in ('2x5','2x10'):
        for profile in ('basic','medium','advanced'):
            keys=[k for k in ordered if k[:2]==(task,profile)]
            a,b=metrics([old[k] for k in keys]),metrics([new[k] for k in keys])
            cells.append(dict(task=task,profile=profile,old=a,new=b,
                completion_delta_pp=100*(b['completion_rate']-a['completion_rate']),
                success_delta_pp=100*(b['success_rate']-a['success_rate'])))
    a,b=metrics(list(old.values())),metrics(list(new.values()))
    payload=dict(protocol=dict(episodes_per_policy=len(ordered),seeds=[args.seed_start,args.seed_start+args.episodes-1],
        source_column=1,steps={'2x5':500,'2x10':3200},official_non_strict=True),
        old=a,new=b,cells=cells,
        completion_delta_pp=100*(b['completion_rate']-a['completion_rate']),
        success_delta_pp=100*(b['success_rate']-a['success_rate']),
        paired_success=dict(new_only=sum(new[k]['success'] and not old[k]['success'] for k in ordered),
            old_only=sum(old[k]['success'] and not new[k]['success'] for k in ordered)),
        paired_completion=dict(improved=sum(new[k]['cookies_in_target']>old[k]['cookies_in_target'] for k in ordered),
            unchanged=sum(new[k]['cookies_in_target']==old[k]['cookies_in_target'] for k in ordered),
            worsened=sum(new[k]['cookies_in_target']<old[k]['cookies_in_target'] for k in ordered)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'paired_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    fmt=lambda v:f'{100*v:.2f}%'
    lines=['# 同种子策略对照','',
        f'每版{len(ordered)}回合，source column 1，2×5为500步、2×10为3200步。完成率为每回合终态入盒比例的平均，成功率采用官方标志。','',
        '| 任务 | 档位 | V5完成率 | V6完成率 | V5成功率 | V6成功率 |',
        '|---|---|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['task']} | {c['profile']} | {fmt(c['old']['completion_rate'])} | {fmt(c['new']['completion_rate'])} | {fmt(c['old']['success_rate'])} | {fmt(c['new']['success_rate'])} |")
    lines+=['',f"整体完成率：{fmt(a['completion_rate'])} → {fmt(b['completion_rate'])}（{payload['completion_delta_pp']:+.2f}个百分点）。",
        f"整体成功率：{fmt(a['success_rate'])}（{a['successes']}/{len(ordered)}）→ {fmt(b['success_rate'])}（{b['successes']}/{len(ordered)}，{payload['success_delta_pp']:+.2f}个百分点）。",'',
        '本次使用新种子，旧版与新版在同一批场景上运行。2×5采用课程非严格判定，可在释放前满足官方成功条件。']
    (args.output/'paired_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:payload[k] for k in ('old','new','completion_delta_pp','success_delta_pp')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
