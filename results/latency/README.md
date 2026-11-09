# Policy computation cost: frozen V10 CPU vs teacher SmolVLA CUDA

## Finding

On these two continuously recorded observation sequences, the teacher model's average native action-call latency is lower than V10's: **70.5 vs 165.0 ms** for 2×5 and **68.7 vs 236.7 ms** for 2×10. The V10/SmolVLA mean latency ratios are **2.34 and 3.45**. This is a synchronous computation benchmark on recorded inputs, not a new closed-loop task-success evaluation.

| Recorded workload | Observations per repeat | V10 mean / median / P95 (ms) | SmolVLA mean / median / P95 (ms) | SmolVLA new-block mean / P95 (ms) | Queue mean (ms) |
|---|---:|---:|---:|---:|---:|
| 2×5 Basic, seed 19000 | 316 | 165.02 / 35.95 / 499.60 | 70.48 / 3.47 / 534.74 | 532.15 / 546.62 | 3.57 |
| 2×10 Advanced, seed 19000 | 671 | 236.69 / 135.52 / 596.55 | 68.69 / 3.52 / 522.95 | 523.49 / 538.54 | 3.61 |

The table is recomputed from the raw per-call CSVs. See `reviewed_summary.json` for unrounded values.

## Measurement setup

The machine has an **Intel Core i5-10500** CPU and **NVIDIA RTX 3090, 24 GiB**. Processes were pinned to CPU cores 0–3, with PyTorch and OpenCV configured to use four threads. The rule policy uses CPU; the teacher model uses CUDA. They ran sequentially, reversing their order for the second of three timing repetitions. Libraries recorded include PyTorch 2.10.0+cu128; the current audit also identified NumPy 2.5.4 and OpenCV 4.14.0. The thread setting describes requested runtime configuration and affinity, not a retrospective measurement of every native-library thread pool.

We measured the complete native `policy.act` call, with inputs already in memory and the returned action available to the CPU. CUDA was synchronized before and after each model call. Image processing, the policy's own geometric calculations, model preprocessing/transfer and output preparation are included. Model loading, PNG decoding, input copying, physics, rendering and external Cartesian-to-joint conversion are excluded. V10 outputs `cartesian_delta`; SmolVLA outputs `joint_position`.

Both use the same original RGB and public robot observations. The V10 files match all **78** entries in the frozen runtime manifest. The teacher checkpoint is `Eter0109/a3-cookie-multi-left8-40k-v1` at revision `ffa4dfbc3e36796a4b02bcf07e6776f6077e49b8`, with SHA256 recorded in `manifest.json`. It runs through the existing course-compatible adapter; the weights are unchanged. Batch size is one, sampling steps 25, generated length 50, and **8 actions are actually consumed before the next block**. The configured internal 512×512 image size and the native caches are retained. The first parameter is float32; that observation is not a claim that every internal operation has the same dtype.

## Why the model has two very different latencies

Generating a new action block takes about **0.53 seconds**. Seven subsequent calls return queued actions and cost about **3.6 ms** each. Complete eight-action cycles average **69.63 ms/action** for 2×5 and **68.58 ms/action** for 2×10. There are 117 and 249 complete cycles, respectively, across the three repetitions. The main all-call average also includes the incomplete final cycle, explaining its small difference from the complete-cycle average.

The rule controller also caches geometry and measurements. Simple movements can be cheap, while observation and geometric checking calls are more expensive. All its long calls remain in the average. In particular, target-retreat calls averaged roughly 5.2/6.1 seconds in the two workloads. This experiment supports separating task reliability from computing cost; it does not support an intrinsic speed advantage for rules over neural policies.

For a 50 ms action period, 57.1%/30.1% of V10 calls and 87.3%/87.5% of model calls finished within the period. The model's percentage is largely due to its queue. These synchronous measurements expose delays rather than certify a real-time or asynchronous controller.

## Input capture and replay review

Both input trajectories were collected once with the frozen V10 and the original task scorer. Their source results were 10/10 and 20/20 with official success. Each actual control step has three original 256×256 RGB images and public state/velocity/end-effector/force data. The capture has 316 and 671 consecutive steps, and 948 and 2,013 PNG images. Capture involves simulation and rendering; **the subsequent timing runs involve neither**. The 2×10 source instruction is 2,8,2,8 cookies. Repeating these trajectories three times measures timing variation, not six independent scenes.

The 2×5 replay's pre- and post-action stages agree with the capture for every step. Full-vector comparison finds a maximum native action-component difference of **0.00834295**, rather than the earlier 0.0055 estimate based only on sampled trace entries.

The 2×10 replay differs at **six pre-action stages per 671-step run** (665/671 = 99.106% match), or 18 differences across all three repetitions. Those frame IDs are 201, 206, 214, 485, 486 and 487. There are also six post-action stage differences per run, and the maximum native action-component difference is **0.598929**. This includes a different movement at frame 410 even though the stage name is the same. These are recorded differences, not an established causal diagnosis of their origin. The replay actually visits grasping, transport, placement and release and does not exhibit the earlier sparse-input observation stall. Timing CSVs did not expose a usable actual batch-index property; source batch indices and the full action/stage comparison are retained separately.

As a sensitivity check, flagging the union of all before/after-stage differences and native action-component differences above 0.01 marks **32 frames** in 2×10. Omitting precisely those frames from both policy timing series changes the means from **236.692/68.691 ms** to **236.764/68.707 ms**. This narrowly checks the influence of directly observed discrepancy frames; it is not a proof about every possible latent-state effect. **All frames remain in the headline table**, and 0.01 is a diagnostic threshold, not a physical safety tolerance.

This is sufficient to report computation costs for these recorded workloads. Exact closed-loop timing, success under the timing runtime, other seeds/profiles, and asynchronous execution are separate experiments. The two earlier 120-episode V10 accuracy tests are unchanged.

## Evidence and independent recomputation

- `raw/`: 12 original, unedited per-call CSVs (2 tasks × 2 policies × 3 repetitions).
- `capture_reference.json`: every captured pre/post stage and simulator-facing native action, plus source-capture metadata. No hidden object positions or private session messages.
- `manifest.json`: run settings, code/checkpoint/input hashes, original metadata hashes and per-run measurements.
- `reviewed_summary.json`: independent recomputation, all-step replay comparisons and sensitivity statistics.
- `code/measure_policy_latency.py`: exact measurement script used; `code/prepare_tape.py`: exact input builder.
- `../../scripts/verify_latency.py`: standard-library-only verifier of the public evidence; it performs no inference, rendering or physical simulation.

From the repository root:

```bash
python3 scripts/verify_latency.py
```

The full image tapes and teacher model weights are not copied into Git. Their hashes and original recipes are supplied for identification. Re-running latency requires those observation tapes (or a separately captured workload) and the pinned policy/model environment; a newly captured workload should be reported as a new measurement rather than assumed to reproduce these exact milliseconds.

The previous sparse-keyframe timing table is superseded; those earlier calls are not included in this dataset. The new interpretation is documented in the paper's computation-cost section.

Timing-method reference: [PyTorch Benchmark Utils](https://docs.pytorch.org/docs/stable/benchmark_utils.html).
