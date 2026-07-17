# Natural small-face relief foundation

The small-face relief path now uses a part-aware Google GNM mean-head prior.
It preserves the detected mouth and limits parametric correction to a softly
feathered nose-and-eye region. This prevents the previous whole-face prior from
flattening expressions while retaining its useful camera-aligned nose and eye
socket geometry.

The final identity-disjoint 20-row gate improves failures `232 -> 229` with
better median shape, gradient, and normalized RMSE and no per-row failure
regression. This is a relative safety improvement, not a production-quality
pass: 229 of 240 absolute part checks still fail, so the new 50% quality floor
keeps the overall gate on hold. The sealed exact gate remains `19 -> 17`; the
15-row varied-scene gate improves `84 -> 82`; and the 30 mm STL remains a
single watertight, winding-consistent component with exact shell agreement and
preserved background depth.

The official Microsoft DAViD Base model was also integrated and CUDA-tested as
a modern human-depth challenger. Its one-row provider smoke passed, but its
eight-row expansion remains on hold because median paired gradient correlation
was slightly lower after fusion.

Full methods, limitations, reproduction commands, metrics, and raw evidence
hashes are published in
`docs/benchmark-evidence/gnm_central_face_foundation_20260717/`.
