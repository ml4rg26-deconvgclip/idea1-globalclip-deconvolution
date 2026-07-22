# Idea 1 V0: globalclip_lysate_noNHS reconstruction

## Goal
Use frozen Parnet-predicted 223 RBP-cell-line binding profiles to reconstruct the observed `globalclip_lysate_noNHS_600bp_signalfiltered.pt` globalCLIP profile with a small interpretable reconstruction head.

## Data flow
1. observed globalCLIP raw counts were normalized into per-window probability profiles
2. real RNA sequences were passed through frozen Parnet
3. Parnet produced 223 predicted profiles per 600 nt window
4. a global softmax reconstruction head learned one non-negative weight per track
5. the weighted mixture was optimized against the observed globalCLIP profile

## Model
For each window and nucleotide position:

pred_global[i, j] = sum_k weight[k] * parnet_profile[i, k, j]

The weights are shared across all windows and constrained with a softmax, so they are non-negative and sum to 1.

## Loss
Profile cross entropy / multinomial-style NLL:

loss = - sum_j observed_profile[i, j] * log(pred_global[i, j] + eps)

## Performance
Validation and test results:

- best validation loss: 5.6061
- test reconstruction head loss: 5.6341
- test unweighted Parnet average loss: 5.8070
- test uniform valid profile loss: 6.3779

Interpretation:
The learned reconstruction head outperforms the unweighted Parnet average and naive baselines, showing that some predicted RBP profiles are much more informative than others for reconstructing the `globalclip_lysate_noNHS` globalCLIP profile.

## Top track-level contributions
Top tracks:

- SND1_K562: 27.62%
- SND1_HepG2: 19.84%
- YBX3_HepG2: 16.68%
- PABPN1_HepG2: 14.96%
- SRSF1_HepG2: 14.54%
- LIN28B_K562: 6.21%

Top 6 tracks explain about 99.85% of the learned mixture.

## Top RBP-level contributions
After aggregating weights across cell lines:

- SND1: 47.46%
- YBX3: 16.68%
- PABPN1: 14.96%
- SRSF1: 14.54%
- LIN28B: 6.21%

Top 5 RBPs explain about 99.85% of the total learned weight.

## Main conclusion
The V0 reconstruction head learns a highly sparse and interpretable mixture. The `globalclip_lysate_noNHS` profile can be reconstructed much better by emphasizing a very small subset of Parnet RBP tracks instead of treating all 223 tracks equally.

## Limitations
- this is a global-weight model, so the same mixture is used for all windows
- track-level attribution does not yet prove causal biological contribution
- highly correlated RBP profiles may make attribution non-identifiable
- control signal is not yet modeled
- peak-level local contribution analysis has not yet been performed

## Minimal next steps
1. use the generated plots in report_assets for presentation
2. optionally inspect a few windows / peaks with top-RBP contributions
3. optionally compare this noNHS lysate run against additional conditions later
