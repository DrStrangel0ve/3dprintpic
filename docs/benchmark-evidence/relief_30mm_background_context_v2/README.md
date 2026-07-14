# 30 mm background context v2 evidence

This directory contains aggregate, privacy-safe evidence for the bounded background-depth follow-up. No private photo, selection mask, render, depth array, or STL is tracked.

- `summary.json` records the exact local RTX 3080 Ti replay metrics for the portrait and llama/group selections.
- The selected subject keeps full scene-inferred depth.
- The mask boundary keeps the slope-bounded support ramp.
- Farther scene context receives a bounded 45% depth budget with a physical feather.
- A final physical-space cap limits far background height after gamma, smoothing, and gradient solves. Attachment bands are clamped from both high and low subject boundaries; mutually incompatible one-pixel constraints are reported separately and cannot be labeled as a strict cap pass.
- Final-surface gates reject collapsed or over-amplified depth, global height shifts, lost gradients, and newly introduced subject-boundary cliffs. A rejected background is restored from the bounded reference and rechecked before export.
- Nonpositive or nonfinite inverse-depth samples cannot poison the selected-depth percentiles.
- Candidate coverage is measured against finite reference context, so missing background pixels fail instead of disappearing from the metric.

The implementation is motivated by gradient-domain bas-relief work that preserves detail after depth compression and explicitly allows non-zero background relief, plus discontinuity-aware surface integration:

- https://people.eecs.berkeley.edu/~sequin/CS285/PAPERS/SIGGRAPH_07/032-weyrich_3D_BasRelief.pdf
- https://doi.org/10.1016/j.cagd.2011.03.003
- https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Discontinuity-preserving_Normal_Integration_with_Auxiliary_Edges_CVPR_2024_paper.pdf
