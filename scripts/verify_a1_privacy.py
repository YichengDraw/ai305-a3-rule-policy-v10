#!/usr/bin/env python3
"""Check the A1 public export for machine-local provenance and verify redacted hashes.

Uses only the Python standard library. If Poppler is installed, also inspect the
rendered report text. This checks A1 files, not historical versions of the repo.
"""
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/a1_regression_20261110'
# Boundary prevents a normal relative package directory named 'data' matching.
LOCAL = re.compile(r'(?<![\w])/(?:Users|home|root|data|workspace|mnt|tmp|private|Volumes)/[^\s\"\']+|[A-Za-z]:\\\\|\\\\[A-Za-z0-9_.-]+\\')
FORBIDDEN_KEYS = {'hostname', 'host_name', 'username', 'user_name', 'LD_LIBRARY_PATH'}
errors = []

def inspect_text(name, text):
    if LOCAL.search(text): errors.append(name + ': absolute local path')
    if name.endswith('.json'):
        def visit(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in FORBIDDEN_KEYS: errors.append(name + ': machine-specific key ' + k)
                    visit(v)
            elif isinstance(value, list):
                for v in value: visit(v)
        visit(json.loads(text))

def inspect_file(path):
    name = path.relative_to(ROOT).as_posix()
    payload = path.read_bytes()
    if name.endswith('.tar.gz'):
        if payload[3] & 8 or payload[4:8] != b'\0'*4: errors.append(name + ': gzip filename or timestamp')
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    errors.append(name + ': unsafe archive member'); continue
                if member.uid or member.gid or member.uname or member.gname or member.mtime:
                    errors.append(name + ': identifying archive metadata')
                inspect_text(name + '#' + member.name, archive.extractfile(member).read().decode())
    elif name.endswith('.gz'):
        if payload[3] & 8 or payload[4:8] != b'\0'*4: errors.append(name + ': gzip filename or timestamp')
        inspect_text(name[:-3], gzip.decompress(payload).decode())
    elif name.endswith('.pdf'):
        if re.search(rb'/(?:CreationDate|ModDate)\b', payload): errors.append(name + ': build-date metadata')
        if shutil.which('pdftotext'):
            inspect_text(name, subprocess.check_output(['pdftotext', str(path), '-'], text=True))
    else:
        inspect_text(name, payload.decode())

paths = [p for folder in (ROOT/'candidates/a1', DATA) for p in folder.rglob('*') if p.is_file()]
paths += [ROOT/p for p in ('README.md', 'report/main.tex', 'report/sections/09_a1.tex', 'report/group3_rule_policy_v10.pdf', 'scripts/run_a1_grid.py', 'scripts/verify_a1.py', 'scripts/verify_a1_privacy.py')]
for path in paths: inspect_file(path)
audit = json.loads((DATA/'sanitization_manifest.json').read_text())
for path, hashes in audit['files'].items():
    if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != hashes['public_sha256']:
        errors.append(path + ': public sanitized hash mismatch')
if errors: raise SystemExit('\n'.join(errors))
print(f'PASS: {len(paths)} public files and all compressed members; no machine-local paths/keys; sanitized hashes match.')
