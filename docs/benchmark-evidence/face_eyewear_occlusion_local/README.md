# Face Eyewear Occlusion Local Replay

This privacy-held replay reproduces the reported 30 mm facial artifact from
cached global and face-crop depth. The personal photograph, face crops, masks,
renders, depth arrays, and STLs are intentionally excluded from git.

The baseline preserved an opaque wraparound lens as a broad depth sheet. The
candidate detects only a broad dark eye-band component, reconstructs its hidden
low-frequency facial surface from the aligned MediaPipe relative-z prior, and
hard-excludes the accepted region from downstream printable detail and bridge
expansion.

Both detected eyewear regions passed all reconstruction gates. Residual error
against the aligned prior fell by `64.86%` and `60.51%`; correction-cap
saturation was `0%` and `0.41%`, below the `5%` maximum. The 30 mm candidate
kept the baseline triangle count and bounding box and remained a single
watertight, consistently wound volume with zero degenerate faces.

Machine-readable aggregate metrics are in `summary.json`.
