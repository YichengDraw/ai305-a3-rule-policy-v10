# Historical implementations and review archive

The final reported policy is `../frozen_policy_v10/`. Historical code below is retained for reproducibility, including failed designs; it is **not** substituted for the final policy.

## Milestones

- `v1`–`v4`: recovered original development snapshots, four Python files per version. These were small-case prototypes, not 120-episode formal releases.
- `v5`–`v9`: verbatim Python files from each original frozen release. Hashes are in each `ARCHIVE_MANIFEST.json` and `versions.json`.
- V10 remains at the repository root in `frozen_policy_v10/`; its original 78-file runtime manifest is unchanged.
- `candidate_archives/`: all retained code snapshots and root Python evaluation/analysis/contract helpers found in the local V5–V13 working directories. Archives through V10 document the report's development history. V11–V13 archives are clearly labeled **post-V10 experiments** and do not change V10's published results. V12 was an audit/selection stage rather than a new frozen policy.

## How to review

Read `../report/group3_rule_policy_v10.pdf` first, then inspect the relevant milestone. For a candidate experiment, consult the matching `candidate_archives/v*_index.json` and extract its tar.gz into a separate directory. Exact hashes preserve provenance. The archives do not claim that every candidate was tested or adopted; adoption is established by the report and its cited result records.

For running a milestone, use the pinned official simulator and pass its directory as `--policy-code`. V1–V5 use `policy:make_policy`, V6–V9 use `policy:make_policy_b`. Keep task budgets and the original factory explicit. Some archived development helpers retain original machine-specific paths; these are historical executable sources, not portable one-command launchers. The current V10 runner is the supported reproducible submission entrypoint.

No user prompt text, chat transcript, secret, private model credential, or unpublished model weights are included. JSON evidence and original native episode extracts are in `../results/history/`. They record actual outcomes including failures, not cherry-picked demonstrations.
