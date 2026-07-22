# ML4RG Idea 1: globalCLIP Deconvolution

This repository contains the Idea 1 code and documented V0 results for using
frozen Parnet predictions to reconstruct observed globalCLIP profiles.

The current V0 model is a small, interpretable reconstruction head. It learns a
single global softmax weight for each of the 223 Parnet RBP-cell-line tracks and
uses the weighted mixture to predict one observed globalCLIP profile per 600 nt
RNA window.

Note on naming: some committed result folders and scripts still use the older
`interphase` run label, for example `docs/idea1_v0_interphase/` and
`results/reconstruction_head/interphase_50`. These are legacy run labels. The
confirmed globalCLIP input for the current V0 analysis is:

```text
globalclip_lysate_noNHS_600bp_signalfiltered.pt
```

## Current Status

The code for the softmax reconstruction head, analysis scripts, result tables,
and report figures is committed to this repository.

The large raw data, pretrained model checkpoints, intermediate `.pt` files, and
full local training outputs are not stored in git. They are expected to live on
the VM or shared storage, under the `twang` workspace. The exact handover paths
should be checked on the VM and recorded before final handoff.

## Key Results

The main result summary is here:

```text
docs/idea1_v0_interphase/RESULTS_SUMMARY.md
```

Despite the legacy directory name, this summary corresponds to the
`globalclip_lysate_noNHS_600bp_signalfiltered.pt` input.

V0 reconstruction performance:

```text
best validation loss:              5.6061
test reconstruction head loss:     5.6341
test unweighted Parnet average:    5.8070
test uniform valid profile:        6.3779
```

The learned softmax mixture is sparse and interpretable. The top track-level
weights are:

```text
SND1_K562:       27.62%
SND1_HepG2:      19.84%
YBX3_HepG2:      16.68%
PABPN1_HepG2:    14.96%
SRSF1_HepG2:     14.54%
LIN28B_K562:      6.21%
```

The top 6 tracks explain about 99.85% of the learned mixture.

## Repository Layout

```text
.
|-- configs/
|   `-- vm.yaml
|-- docs/
|   |-- idea1_v0_interphase/
|   |   |-- RESULTS_SUMMARY.md
|   |   |-- evaluation_metrics.csv
|   |   |-- training_metrics.csv
|   |   |-- reconstruction_weights_rbp_level.csv
|   |   |-- reconstruction_weights_track_level_annotated.csv
|   |   |-- top30_track_weights_annotated.csv
|   |   `-- report_assets/
|   `-- vm_parnet_environment_notes.md
|-- figures/
|   |-- baseline_comparison_test.pdf
|   |-- baseline_comparison_test.png
|   |-- top_rbp_weights.pdf
|   `-- top_rbp_weights.png
|-- report_assets/
|   `-- step1_interphase/
|-- scripts/
|   |-- 07_prepare_globalclip_profiles.py
|   |-- 08_run_parnet_on_globalclip_windows.py
|   |-- train_reconstruction_head.py
|   |-- evaluate_profile_correlations.py
|   |-- export_per_window_profile_correlations.py
|   |-- analysis/
|   |-- env/
|   `-- report/
|-- .gitignore
`-- README.md
```

## Data Flow

The V0 pipeline is:

1. Convert observed globalCLIP raw counts into per-window probability profiles.
2. Run frozen Parnet on the matching RNA sequence windows.
3. Save 223 predicted Parnet profiles per 600 nt window.
4. Train a global softmax reconstruction head over the 223 Parnet tracks.
5. Compare the reconstruction head against simple baselines.
6. Export result tables and report figures.

For a window `i`, nucleotide position `j`, and Parnet track `k`:

```text
pred_global[i, j] = sum_k weight[k] * parnet_profile[i, k, j]
```

The weights are shared across all windows. They are constrained by a softmax, so
they are non-negative and sum to 1.

The training loss is profile cross entropy / multinomial-style NLL:

```text
loss = -sum_j observed_profile[i, j] * log(pred_global[i, j] + eps)
```

## Important Files

Core pipeline scripts:

```text
scripts/07_prepare_globalclip_profiles.py
scripts/08_run_parnet_on_globalclip_windows.py
scripts/train_reconstruction_head.py
```

Analysis and reporting scripts:

```text
scripts/evaluate_profile_correlations.py
scripts/export_per_window_profile_correlations.py
scripts/plot_profile_correlation_metrics.py
scripts/analysis/
scripts/report/
```

Step 1 exploratory outputs:

```text
report_assets/step1_interphase/
```

The older `scripts/evaluate_reconstruction.py` is still a placeholder. Use the
newer evaluation and reporting scripts listed above for the current V0 analysis.

## Running The Pipeline

Run these commands on the VM or in an environment with access to the raw
globalCLIP data, pretrained Parnet checkpoint, and Parnet dependencies.

Prepare observed target profiles:

```bash
python3 scripts/07_prepare_globalclip_profiles.py \
  --input-pt <path_to>/globalclip_lysate_noNHS_600bp_signalfiltered.pt \
  --output-dir results/globalclip_profiles \
  --condition globalCLIP \
  --output-name globalclip_lysate_noNHS_target_profiles.pt
```

Run frozen Parnet on the same windows:

```bash
python3 scripts/08_run_parnet_on_globalclip_windows.py \
  --input-pt <path_to>/globalclip_lysate_noNHS_600bp_signalfiltered.pt \
  --config configs/vm.yaml \
  --output-dir results/parnet_predictions \
  --condition-name globalclip_lysate_noNHS \
  --device auto \
  --output-name globalclip_lysate_noNHS_parnet_predictions.pt
```

Train the softmax reconstruction head:

```bash
python3 scripts/train_reconstruction_head.py \
  --parnet-predictions results/parnet_predictions/globalclip_lysate_noNHS_parnet_predictions.pt \
  --target-profiles results/globalclip_profiles/globalclip_lysate_noNHS_target_profiles.pt \
  --output-dir results/reconstruction_head/globalclip_lysate_noNHS \
  --epochs 100 \
  --batch-size 64 \
  --device auto
```

The training script writes:

```text
checkpoint.pt
training_metrics.csv
evaluation_metrics.csv
reconstruction_weights_track_level.csv
metadata.json
```

## VM And Storage Notes

Machine-specific paths should live in:

```text
configs/vm.yaml
```

The repository intentionally ignores heavy local files:

```text
data/
models/
results/
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.tar.gz
vm_audit_*.txt
```

Known VM note currently recorded in `docs/vm_parnet_environment_notes.md`:

```text
~/storage_ml4rg26-deconvgclip/twang/repos/parnet--demo--train-models
```

Before final handoff, confirm and document the exact VM/shared-storage locations
for:

```text
raw globalCLIP dataset
confirmed condition/input label: globalclip_lysate_noNHS
pretrained Parnet checkpoint
prepared target profiles
saved Parnet predictions
full reconstruction-head training outputs
any archived result bundles
```

## Handoff Summary

For collaborators:

```text
Code and V0 softmax-head results:
ml4rg26-deconvgclip/idea1-globalclip-deconvolution

Result summary:
docs/idea1_v0_interphase/RESULTS_SUMMARY.md

Figures:
figures/
docs/idea1_v0_interphase/report_assets/

Heavy data and model artifacts:
not in git; check the VM/shared storage under twang

Confirmed V0 raw input file:
globalclip_lysate_noNHS_600bp_signalfiltered.pt
```

## Limitations And Next Steps

Current limitations:

1. The V0 model uses one global mixture for all windows.
2. Track-level attribution does not prove causal biological contribution.
3. Correlated RBP profiles may make attribution non-identifiable.
4. The control signal is not modeled yet.
5. Peak-level local contribution analysis is not complete yet.

Useful next steps:

1. Confirm the exact VM/shared-storage paths and add them to this README.
2. Package or document the full input/intermediate/output files for handoff.
3. Inspect selected windows or peaks with high top-RBP contributions.
4. Optionally clean up historical `interphase` run labels in file and directory
   names once doing so will not break existing references.
5. Extend the model beyond a single global softmax mixture if a richer follow-up
   analysis is needed.
