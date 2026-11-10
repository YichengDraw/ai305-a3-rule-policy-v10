#!/usr/bin/env python3
"""Verify A1 code, original terminal records, initializations and paired metrics offline.

No policy is imported, no simulation runs, and no reported success is redefined.
"""
import ast
import csv
import gzip
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/a1_regression_20261110'
CODE = ROOT / 'candidates/a1'
def digest(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text())
def key(r): return r['task'], r['profile'], int(r['seed'])
def check(c, message):
    if not c: raise ValueError(message)

redaction = read(DATA / 'sanitization_manifest.json')
for relative, recorded in redaction['files'].items():
    check(digest((ROOT / relative).read_bytes()) == recorded['public_sha256'], f'Sanitized artifact hash mismatch: {relative}')

manifest = read(CODE / 'manifest.json')
hashes = manifest['runtime_sha256']
check(len(hashes) == 79, 'Expected 79 runtime files')
check({str(p.relative_to(CODE)) for p in CODE.rglob('*.py')} == set(hashes), 'Unexpected runtime file set')
for path, expected in hashes.items():
    payload = (CODE / path).read_bytes()
    check(digest(payload) == expected, f'Runtime hash mismatch: {path}')
    ast.parse(payload.decode())
node = ast.parse((CODE / 'policy.py').read_text())
deps = next(ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'RUNTIME_DEPENDENCIES' for t in n.targets))
check(set(deps) == set(hashes), 'Runtime dependency declaration mismatch')
fingerprint = digest(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode())
check(fingerprint == manifest['runtime_fingerprint_sha256'], 'Runtime fingerprint mismatch')

native = read(DATA / 'result_manifest.json')
check(digest((DATA / native['native_archive']).read_bytes()) == native['archive_sha256'], 'Archive hash mismatch')
with tarfile.open(DATA / native['native_archive']) as archive:
    check(set(archive.getnames()) == set(native['members_sha256']), 'Native archive membership mismatch')
    raw = {}
    for member, expected in native['members_sha256'].items():
        payload = archive.extractfile(member).read()
        check(digest(payload) == expected, f'Native result hash mismatch: {member}')
        row = json.loads(payload)
        check(key(row) not in raw, 'Duplicate native result')
        raw[key(row)] = row
check(len(raw) == 240, 'Expected 240 native results')
with (DATA / 'episodes.csv').open(newline='') as f: exported = list(csv.DictReader(f))
check(len(exported) == 240 and len({key(r) for r in exported}) == 240, 'Incomplete CSV')
for r in exported:
    a = raw[key(r)]
    check(int(r['cookies_in_target']) == a['cookies_in_target'] == a['score'], 'Official terminal count mismatch')
    check(int(r['capacity']) == a['max_score'] == (10 if a['task'] == '2x5' else 20), 'Capacity mismatch')
    check((r['official_success'] == 'true') == a['success'], 'Official success mismatch')
    check(int(r['control_steps']) == a['steps'], 'Terminal step mismatch')
    check(a['source_column'] == 1 and a['success_hold_steps'] == 5 and a['strict_policy_completion'] is False, 'Official protocol mismatch')
    check(a['max_steps'] == (500 if a['task'] == '2x5' else 3200), 'Step budget mismatch')
    check(abs(float(r['completion_rate']) - a['score']/a['max_score']) < 1e-12, 'Completion ratio mismatch')

# Selected records may come from a root containing other failed/invalid attempts.
# Check each selected job rather than equating root completion with task success.
protocols = {}
for i in range(6):
    p = read(DATA / f'protocols/root{i}_protocol.json')
    completed = read(DATA / f'protocols/root{i}_complete.json')
    check({k:p['code_sha256'][k] for k in hashes} == hashes, f'Protocol {i} runtime differs')
    check(p['success_hold_steps'] == 5 and p['strict_policy_completion'] is False, 'Root scoring mismatch')
    check(p['runtime_truth_access'] is False and p['official_scoring_changed'] is False, 'Root policy/scorer boundary changed')
    check(p['rendering']['backend'] == 'osmesa' and p['rendering']['device'] == 'CPU', 'Rendering mismatch')
    protocols[i] = {j['label']: j for j in completed['jobs']}
for p in native['provenance']:
    label = p['source_relative_path'].split('/')[0]
    check(protocols[p['source_root_id']][label]['returncode'] == 0, 'Selected run has nonzero exit')
    check(native['members_sha256'][p['archive_member']] == p['result_sha256'], 'Provenance hash mismatch')

# V10 baseline CSVs are unchanged existing public release records.
baseline = {}
for path in [ROOT/'results/v10_holdout_episodes.csv', ROOT/'results/retest/episodes.csv']:
    with path.open(newline='') as f:
        for r in csv.DictReader(f): baseline[key(r)] = r
check(set(baseline) == set(raw), 'A1/V10 episode keys do not match')
initial = json.loads(gzip.decompress((DATA/'baseline_initializations.json.gz').read_bytes()))
check(len(initial['rows']) == 240 and len({key(r) for r in initial['rows']}) == 240 and {key(r) for r in initial['rows']} == set(raw), 'Incomplete or duplicate baseline initializations')
terminal_keys = set(initial['omitted_terminal_keys'])
check(terminal_keys == {'cookies_in_source','cookies_in_target','final_expert_phase','grasp_reports','target_column_counts'}, 'Unexpected omitted initialization fields')
mode_only = 0
for ref in initial['rows']:
    a = raw[key(ref)]
    actual = {k:v for k,v in a['randomization'].items() if k not in terminal_keys}
    source = ref['randomization']
    different = {k for k in actual.keys() | source.keys() if actual.get(k) != source.get(k)}
    if different:
        check(a['task'] == '2x10' and 24000 <= a['seed'] <= 24019 and different == {'first_grasp_mode'}, 'Non-equivalent initialization')
        sampled = random.Random(f"a3-cookie-2x10-first-grasp-v1:{a['seed']}").randrange(10)
        check(actual['first_grasp_count'] == source['first_grasp_count'] == sampled == int(baseline[key(a)]['first_grasp_n']), 'Fixed grasp count differs from original random draw')
        mode_only += 1
    check(a['task_instruction'] == ref['task_instruction'], 'Task instruction mismatch')
    for k in ('source_column','max_steps','success_hold_steps','strict_policy_completion'):
        check(a[k] == ref[k], f'Baseline protocol mismatch: {k}')
    if a['seed'] < 620000:
        check(ref['original_sha256'] == baseline[key(a)]['original_episode_sha256'], 'Original baseline hash mismatch')
check(mode_only == 60, 'Expected precisely 60 metadata-only fixed-mode rows')
# The separately published V10 replication provides an independent full native source.
original_retest = gzip.decompress((ROOT/'results/retest/native_episodes.jsonl.gz').read_bytes())
retest = {key(r):r for r in map(json.loads, original_retest.splitlines())}
for ref in initial['rows']:
    if key(ref) in retest:
        check(ref['original_sha256'] == digest(original_retest), 'Retest baseline source hash mismatch')
        check(ref['randomization'] == {k:v for k,v in retest[key(ref)]['randomization'].items() if k not in terminal_keys}, 'Retest baseline extraction mismatch')

# Check pinned upstream code used in the fixed/random mode explanation.
environment = read(DATA/'environment_snapshot.json')
upstream = DATA/'upstream_mode_evidence'
for source in upstream.rglob('*.py'):
    relative = source.relative_to(upstream).as_posix()
    check(digest(source.read_bytes()) == environment['source_files_sha256'][relative], f'Upstream evidence hash mismatch: {relative}')
check(len(list(upstream.rglob('*.py'))) == 3, 'Missing mode-evidence source')

summary = read(DATA/'summary.json')
for grid in ('grid24000','grid620xxx'):
    rows = [r for r in exported if r['grid'] == grid]
    check(len(rows) == 120, 'Incomplete grid')
    cells = Counter((r['task'],r['profile']) for r in rows)
    check(len(cells) == 6 and set(cells.values()) == {20}, 'Unbalanced grid')
    completion = sum(int(r['cookies_in_target'])/int(r['capacity']) for r in rows)/120
    successes = sum(r['official_success'] == 'true' for r in rows)
    check(abs(completion-summary['grids'][grid]['completion_rate']) < 1e-12 and successes == summary['grids'][grid]['successes'], 'Summary mismatch')
    for c in [c for c in summary['cells'] if c['grid'] == grid]:
        cell = [r for r in rows if r['task'] == c['task'] and r['profile'] == c['profile']]
        check(len(cell) == c['n'] == 20, 'Cell size mismatch')
        check(sum(int(r['cookies_in_target']) for r in cell) == c['cookies'], 'Cell total mismatch')
        check(abs(sum(int(r['cookies_in_target'])/int(r['capacity']) for r in cell)/20-c['completion_rate']) < 1e-12, 'Cell completion mismatch')
        check(sum(r['official_success'] == 'true' for r in cell) == c['successes'], 'Cell success mismatch')
    gained = [key(r) for r in rows if r['official_success'] == 'true' and baseline[key(r)]['official_success'] != 'true']
    lost = [key(r) for r in rows if r['official_success'] != 'true' and baseline[key(r)]['official_success'] == 'true']
    regressed = [key(r) for r in rows if int(r['cookies_in_target']) < int(baseline[key(r)]['cookies_in_target'])]
    check(not lost and not regressed, 'Published no-count/success-regression claim failed')
    check(len(gained) == (0 if grid == 'grid24000' else 3), 'Unexpected gained-success count')
    print(f'{grid}: completion={completion:.6%}; success={successes}/120; gained={gained}')
check(abs(sum(a['score']/a['max_score'] for a in raw.values())/240-summary['pooled']['completion_rate']) < 1e-12 and sum(a['success'] for a in raw.values()) == summary['pooled']['successes'], 'Pooled A1 result mismatch')
# W1b is an auxiliary terminal-summary comparison; its full native logs are not bundled.
with (DATA/'paired_results.csv').open(newline='') as f: paired = list(csv.DictReader(f))
check(len(paired) == 240 and len({key(r) for r in paired}) == 240 and {key(r) for r in paired} == set(raw), 'Incomplete paired summary')
final = read(DATA/'FINAL_VERIFICATION.json')
check(not final['issues'], 'Final source audit reports issues')
for r in paired:
    k = key(r)
    check(int(r['A1_count']) == raw[k]['cookies_in_target'] and (r['A1_success'] == 'True') == raw[k]['success'], 'Paired A1 summary mismatch')
    check(int(r['V10_count']) == int(baseline[k]['cookies_in_target']) and (r['V10_success'] == 'True') == (baseline[k]['official_success'] == 'true'), 'Paired V10 summary mismatch')
    check(not (r['W1b_success'] == 'True' and r['A1_success'] != 'True') and int(r['A1_count']) >= int(r['W1b_count']), 'A1/W1b regression claim failed')
for grid in ('grid24000','grid620xxx'):
    rows = [r for r in paired if (int(r['seed']) < 620000) == (grid == 'grid24000')]
    for name in ('A1','W1b'):
        c = sum(int(r[name+'_count'])/(10 if r['task'] == '2x5' else 20) for r in rows)/120
        n = sum(r[name+'_success'] == 'True' for r in rows)
        source = final['policies'][name]['summary'][grid]
        check(abs(c-source['C']) < 1e-12 and n == source['successes'], f'{name} source-audit summary mismatch')
        if name == 'W1b':
            check(abs(c-summary['W1b'][grid]['C']) < 1e-12 and n == summary['W1b'][grid]['successes'], 'W1b public summary mismatch')
print(f'PASS: 79 runtime hashes; 240 original results; 240 initializations/task texts; 60 metadata-only fixed-mode replays; fingerprint={fingerprint}')
