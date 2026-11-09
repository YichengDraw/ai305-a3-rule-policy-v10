#!/usr/bin/env python3
"""Recompute published latency and replay differences from raw CSVs; no model/simulator."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag(value):
    if value not in ('True', 'False', 'true', 'false'):
        raise ValueError('Invalid boolean: ' + str(value))
    return value.lower() == 'true'


def stats(values):
    a = sorted(values)
    assert a and all(math.isfinite(x) and x >= 0 for x in a)
    rank = .95 * (len(a) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return {'n': len(a), 'mean_ms': statistics.mean(a), 'median_ms': statistics.median(a),
            'p95_ms': a[lo] + (a[hi] - a[lo]) * (rank - lo), 'max_ms': a[-1],
            'fraction_under_50ms': sum(x <= 50 for x in a) / len(a)}


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for x in rows:
        for k in ('round', 'episode', 'call', 'frame_id', 'action_dim'):
            x[k] = int(x[k])
        for k in ('finished_before', 'finished_after', 'new_chunk'):
            x[k] = flag(x[k])
        x['latency_ms'] = float(x['latency_ms'])
        x['action'] = json.loads(x['action_json'])
        assert len(x['action']) == x['action_dim']
        assert all(math.isfinite(v) for v in x['action'])
    return rows


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    reference = json.loads((root / 'capture_reference.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert sha(root / name) == digest, 'Evidence hash mismatch: ' + name
    result = {'scope': manifest['study'], 'tasks': {},
              'review_decision': 'Report as synchronous recorded-workload computation, not exact closed-loop execution',
              'warnings': ['2x10 has six phase differences per replay, not 18 out of 671.',
                           'Actions were compared on every captured step; phase equality is not action equality.',
                           'Mean, P95 and deadline counts describe these two workloads, not all scenes.']}
    for task in ('2x5', '2x10'):
        capture = reference['tasks'][task]
        refs = capture['reference']; n = len(refs)
        assert n == capture['steps'] and capture['success']
        assert capture['code_mismatches'] == [] and capture['runtime_file_count'] == 78
        assert capture['png_count'] == 3*n
        assert [x['frame_id'] for x in refs] == list(range(n))
        assert capture['recording']['public_stride'] == capture['recording']['lossless_stride'] == 1
        groups = {'v10': [], 'smolvla': []}; rounds = []; cycles = []; action_patterns = []
        sensitivity_sets = []
        for repeat in (1, 2, 3):
            pair = {}
            for pol, device in (('v10', 'cpu'), ('smolvla', 'cuda')):
                path = root / 'raw' / f'{task}_{pol}_{device}_r{repeat}.csv'
                rows = read_rows(path)
                assert len(rows) == n and [x['frame_id'] for x in rows] == list(range(n))
                assert all(x['round'] == repeat and x['task'] == task and x['policy'] == pol for x in rows)
                assert all(x['episode'] == 0 for x in rows)
                assert not any(x['finished_before'] for x in rows)
                assert [x['reference_phase'] for x in rows] == [x['phase_before'] for x in refs]
                meta = next(m for m in manifest['runs'] if m['task'] == task and m['round'] == repeat and m['policy'] == pol)
                assert meta['simulation_steps'] == meta['render_calls'] == 0
                assert meta['affinity'] == [0, 1, 2, 3] and meta['threads'] == 4
                recomputed = stats([x['latency_ms'] for x in rows])
                for k in ('mean_ms', 'median_ms', 'p95_ms', 'max_ms'):
                    assert abs(recomputed[k] - meta['all_calls'][k]) < 1e-7, (task, pol, repeat, k)
                pair[pol] = rows; groups[pol].extend(rows)
                if pol == 'smolvla':
                    assert (meta['n_action_steps'], meta['num_steps'], meta['chunk_size']) == (8,25,50)
                    starts = [i for i,x in enumerate(rows) if x['new_chunk']]
                    assert starts == list(range(0,n,8))
                    for i in starts:
                        z = rows[i:i+8]
                        if len(z) == 8:
                            assert z[0]['new_chunk'] and not any(x['new_chunk'] for x in z[1:])
                            cycles.append(sum(x['latency_ms'] for x in z)/8)
            assert next(m['tape_sha256'] for m in manifest['runs'] if m['task']==task and m['round']==repeat and m['policy']=='v10') == next(m['tape_sha256'] for m in manifest['runs'] if m['task']==task and m['round']==repeat and m['policy']=='smolvla')
            a = pair['v10']
            before = [i for i,(x,y) in enumerate(zip(a,refs)) if x['policy_phase_before'] != y['phase_before']]
            after = [i for i,(x,y) in enumerate(zip(a,refs)) if x['policy_phase_after'] != y['phase_after']]
            errs = [max(abs(u-v) for u,v in zip(x['action'], y['action'])) for x,y in zip(a,refs)]
            assert all(len(x['action']) == len(y['action']) for x,y in zip(a,refs))
            # Diagnostic sensitivity, not an acceptance threshold or a replacement headline.
            flagged = set(before) | set(after) | {i for i,e in enumerate(errs) if e > .01}
            sensitivity_sets.append(flagged)
            rounds.append({'repeat': repeat, 'steps': n,
                'v10_mean_ms': statistics.mean(x['latency_ms'] for x in a),
                'smolvla_mean_ms': statistics.mean(x['latency_ms'] for x in pair['smolvla']),
                'phase_before_mismatch_frames': before, 'phase_after_mismatch_frames': after,
                'phase_before_match_fraction': (n-len(before))/n,
                'max_absolute_native_action_difference': max(errs),
                'action_difference_over_0_01_count': sum(e > .01 for e in errs),
                'batch_index_recorded_by_timing': all(x.get('actual_batch_index') not in (None,'','UNKNOWN') for x in a),
                'sensitivity_flagged_frames': sorted(flagged)})
            action_patterns.append([(x['policy_phase_before'], x['policy_phase_after'], x['action']) for x in a])
        assert action_patterns[0] == action_patterns[1] == action_patterns[2]
        assert sensitivity_sets[0] == sensitivity_sets[1] == sensitivity_sets[2]
        rule, model = groups['v10'], groups['smolvla']
        flagset = sensitivity_sets[0]
        rs = stats([x['latency_ms'] for x in rule]); ms = stats([x['latency_ms'] for x in model])
        sensitivity = {pol: stats([x['latency_ms'] for x in rows if x['frame_id'] not in flagset])
                       for pol,rows in groups.items()}
        result['tasks'][task] = {'profile': capture['profile'], 'seed': capture['seed'],
            'source_column': capture['source_column'], 'frames_per_repeat': n,
            'actual_v10_phase_counts': dict(Counter(x['policy_phase_before'] for x in rule)),
            'reference_batch_counts': dict(Counter(str(x['batch_index_before']) for x in refs)),
            'repeated_state_action_sequence_identical': True,
            'v10': rs, 'smolvla_all_calls': ms,
            'smolvla_refresh': stats([x['latency_ms'] for x in model if x['new_chunk']]),
            'smolvla_queue': stats([x['latency_ms'] for x in model if not x['new_chunk']]),
            'full_eight_action_cycles': len(cycles), 'full_cycle_amortized_ms_per_action': statistics.mean(cycles),
            'v10_to_smolvla_mean_latency_ratio': rs['mean_ms']/ms['mean_ms'], 'rounds': rounds,
            'sensitivity': {'purpose':'Omit direct phase/action-difference frames from BOTH timings only as a sensitivity check; main table keeps all frames.',
                'native_action_cutoff_for_flagging': .01, 'flagged_frames_per_repeat':len(flagset),
                'retained_means':{k:v['mean_ms'] for k,v in sensitivity.items()},
                'v10_to_smolvla_ratio':sensitivity['v10']['mean_ms']/sensitivity['smolvla']['mean_ms']}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'results/latency')
    ap.add_argument('--write-summary', type=Path)
    args = ap.parse_args()
    result = analyze(args.root)
    if args.write_summary:
        args.write_summary.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    else:
        saved=json.loads((args.root/'reviewed_summary.json').read_text())
        assert saved == result, 'Published reviewed_summary differs from raw recomputation'
    for task,r in result['tasks'].items():
        print(f"{task}: {r['frames_per_repeat']} frames x3; V10 {r['v10']['mean_ms']:.3f} ms; SmolVLA {r['smolvla_all_calls']['mean_ms']:.3f} ms; latency ratio {r['v10_to_smolvla_mean_latency_ratio']:.3f}; before-phase differences per repeat {len(r['rounds'][0]['phase_before_mismatch_frames'])}")
    print('PASS: raw hashes, paired inputs, frame continuity, CUDA-chunk cadence and published calculations verified.')


if __name__ == '__main__':
    main()
