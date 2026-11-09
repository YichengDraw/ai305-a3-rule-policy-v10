# Group 3 research report — advisor–executor rule policy, V1–V10

**PDF:** [group3_rule_policy_v10.pdf](group3_rule_policy_v10.pdf)

**Authors:** Yicheng Zhao, Sihua Zhang, Ruilin Gao (Group 3).

The nine-page, two-column paper describes the A3 cookie-transfer task, the explicit camera/force-based rule policy, ten implementation milestones, and the AI advisor–executor development loop. It includes real same-seed comparisons, the original frozen V10 holdout results and a separate 20-seed replication. All code versions and experimental records linked by the paper are in this repository. The paper does not reproduce private agent prompts.

**Read in this order:** the PDF; [final code](../frozen_policy_v10/); [historical versions](../history/README.md); [first test](../results/v10_official_grid_summary.json); [independent replication](../results/retest/README.md). The public assignment's source column 1 is used in both numerical headline tables.

## Rebuilding the paper

The editable LaTeX entrypoint is `main.tex`; the content is in `sections/*.tex`. Figures are included as PNG files under `figures/`. Run either of the following from the `report/` folder with a standard TeX Live installation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Or use `tectonic main.tex`. The optional `make_figures.py` generates the three original diagrams from the saved history table (requires Pillow); `build_evidence_tables.py` reconstructs the appendix table from `results/history/formal_history_summary.json`.

For result integrity, use `python3 scripts/verify_release.py` and `python3 scripts/verify_retest.py` from the repository root. Neither command runs a simulator.
