# MHR bounded continuous low-rank audit

This diagnostic study asks a deliberately narrow question: does a bounded,
continuous search over pinned MHR identity, expression, camera, and fusion
parameters find a strict improvement over the current face-depth incumbent on
the hardest 74-75-pixel turned face?

No winner was observed in the measured 640 candidates. Production remains
unchanged.

## Research and method

The earlier finite MHR sample sweep did not test the continuous coefficient
space. This follow-up uses SciPy's bounded differential-evolution optimizer
with deterministic Sobol initialization. It evaluates the real exact six-part
shape, raw-gradient, and affine-mm metrics at every candidate. The four-
generation optimizer budget ended without convergence, so this is a bounded
sampled audit, not a global upper-bound claim.

The evaluator reuses the checksum-verified public
[official MHR](https://github.com/facebookresearch/MHR) model at source revision
`4998cec385b1aaa07abdefba71bfba2f83c7db32`. The MHR source and model are
Apache-2.0. The search procedure follows the bounded population semantics in
the official [SciPy differential evolution documentation](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.optimize.differential_evolution.html).

All bases are fit from the authoritative 240-row training partition. The
manifest row IDs, split labels, and identity groups must match the corpus
summary exactly. Validation and sealed identities are excluded.

The experiment is explicitly pose-oracle diagnostic evidence. Candidate
generation is centered on the fixture's ground-truth camera yaw and elevation,
which are unavailable to the production path. It is never promotion eligible.
Schema v2 verifies the SHA256 and size of every declared source, mask,
ground-truth, request, response, and face-part asset; content-addresses every
consumed job depth/mask file; records the clean pinned MHR checkout preflight;
and hashes the running evaluator implementation.

Plain coefficient PCA retained too little expression variation for a useful
rank-24 search. The final representation instead builds the MHR blendshape
displacement Jacobian over 5,582 head vertices and weights it by surface area
plus six equal-total-weight facial regions. Rank 12 identity and rank 34
expression retain `0.974423` and `0.992261` global weighted geometry variance.
The worst six-region retentions are `0.950358` and `0.957279`. Rank 32
expression failed closed at `0.946306` right-eye retention, below the `0.95`
gate.

## Hard-row result

The authoritative schema-v2 search evaluated 640 valid candidates with no
inference failures in `490.61 s` and `1.105 GiB` measured peak VRAM on the local
RTX 3080 Ti. The
candidate background remained bit-exact.

| Method | Failures | Shape | Raw gradient | Affine RMSE |
| --- | ---: | ---: | ---: | ---: |
| Raw baseline | 10 | 0.804774 | 0.625533 | 0.176007 |
| Production incumbent | 9 | 0.807012 | 0.625799 | 0.175104 |
| Best observed candidate | 9 | 0.805054 | 0.625820 | 0.175894 |

The best candidate strictly beats the raw baseline, but only ties the
incumbent's failure count. It regresses shape correlation and affine RMSE and
introduces a nose shape failure that the incumbent does not have. The strict
incumbent gate therefore has no winner.

## Decision

Hold this bounded low-rank configuration. A tied discrete failure count is not
accepted in exchange for worse global face shape, worse metric depth error, or
a newly failed named part. Because no candidate passed the exact incumbent
gate, no 30 mm physical replay was run; spending that replay would not change
the decision.

The production face/background path and its existing 30 mm background, cap,
attachment, topology, shell, object, llama, dark-skin, eyewear, and cast-shadow
controls remain unchanged. The next experiment should work upstream with a
maintained camera-aligned geometry provider or a genuinely trainable image-to-
geometry representation, rather than continue exact-row coefficient tuning.

Machine-readable metrics, run hashes, and provenance are in `results.json`.
