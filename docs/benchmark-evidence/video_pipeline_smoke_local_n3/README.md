# Video Pipeline Smoke Evidence

This deterministic synthetic turntable slice checks video decoding, frame selection, temporal object masks,
turntable-camera assignment, visual-hull reconstruction, printable STL repair, and ground-truth surface distance.

| Frame selector | Samples | Mask IoU median | Mask IoU minimum | Chamfer L1 median | H95 median | STL gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sharpness-motion-selector | 3 | 0.9991 | 0.9939 | 0.0605 | 0.1259 | pass |
| uniform-frame-sampler | 3 | 0.9989 | 0.9925 | 0.0338 | 0.0833 | pass |

Overall gate: **pass**.
Recommended frame selector: **`uniform-frame-sampler`**, chosen by median final-mesh Chamfer among passing lanes.

The built-in turntable lane is a controlled-capture baseline. SAM 2 video, SAM 3.1, and VGGT-Omega
readiness is recorded in `results.json`; unavailable dependencies or gated checkpoints are reported as
setup blockers and are not counted as successful model runs.

## Learned Segmentation Smoke

`sam2_tiny_cpu_smoke.json` records a real pinned `facebook/sam2.1-hiera-tiny` run at revision
`de431c4043854a71d8101e17995dfe596bf101a5`. Three propagated 256 px masks passed every temporal gate,
with median ground-truth IoU `0.99991`, minimum IoU `0.99973`, and CPU runtime `23.295` seconds.
