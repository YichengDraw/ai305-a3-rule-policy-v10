# A1: separately frozen development candidate

This is the exact **79-file A1 runtime snapshot** evaluated in the November 10 paired regression. It does not replace `frozen_policy_v10/` or change the formal submission.

- Factory: `policy:make_policy_b`; action mode: `cartesian_delta`.
- Runtime fingerprint: `a1f7f3e2d2a52b061b81dd72c2a566d2b25993bf60eeeb729f236a1a98f82e57` (canonical path-to-SHA256 map, defined in `manifest.json`).
- All runtime bytes match each of the six source-run protocols. The filenames retain their historical `v10_leaves_...` namespace deliberately.
- [240 native records, method, limitations and verifier](../../results/a1_regression_20261110/README.md).

## Changes from frozen V10

The substantive changes are confined to the source-column-1 2×10 branch: source-frontier proposal selection in `perception.py`, and camera-order-independent occupied/free conflict handling in `target_occupancy.py` plus the new `fusion_conflict.py`. `policy.py` in that branch adds one diagnostic field; the root `policy.py` adds the new runtime dependency. Other runtime files are byte-identical to V10.

A1 keeps the parent's precise source-frontier estimate when available and permits the alternative selection after its observation-prefix gate when the parent is unresolved. This is the tested implementation, not a claim that every observation-history entry is a valid completed observation. Known hard cases and future gate validation remain development work. The target fusion rejects conflicting occupied/free evidence rather than letting camera order decide what overwrites stored occupancy.

This snapshot is not a timing-optimization candidate, and no A1 latency advantage is claimed. Do not mix it with the separate efficiency experiments or later transport-force variants.
