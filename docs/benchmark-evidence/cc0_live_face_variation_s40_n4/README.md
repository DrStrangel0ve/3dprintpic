# CC0 live face variation evidence

This directory contains compact, privacy-safe evidence for the 30 mm face and
background relief regression. Raw source renders, masks, depth arrays, API
responses, and STL files remain under the clean checkout's ignored
`backend/output` directory.

- Exact implementation and producer revision: `4549b28eae7864799ab4adb0fc4d4a7499fae99b`.
- Four paired live API rows passed: 256 px small/off-axis/yawed, 384 px neutral,
  384 px turned with an eye-band occluder, and 384 px close/turned.
- Every row validated one human face. Generic selected-component fallback does
  not satisfy the face gate.
- Refined provider depth is compared with retained CC0 rendered depth using
  affine-aligned shape correlation, two-axis gradient correlation, normalized
  RMSE, coverage, and depth-semantics orientation.
- The small 31 px mapped landmark face is recovered by selection-ROI upscaling.
- The occluded row uses YuNet detection followed by a face-specific MediaPipe
  crop. The connected eyewear band is detected; reconstruction is skipped only
  when the measured source/prior residual is already below the 5% artifact gate.
- Candidate background detail is paired against a zero-detail control. Face
  interior and attachment changes remain bounded while background detail stays
  source-aligned.
- Every candidate passes background retention, physical cap, attachment,
  watertight/manifold/single-component topology, and exact STL shell checks.
- Final local validation passed 596 backend tests plus 72 subtests.

The detector configuration follows the official MediaPipe Face Landmarker
confidence controls and OpenCV FaceDetectorYN score/NMS controls:

- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions
- https://docs.opencv.org/master/df/d20/classcv_1_1FaceDetectorYN.html

`summary.json` is a compact transcription of the tracked measurements. The
ignored full summary SHA256 is
`6e529aa41dfe93b9af95b73207e0114a325cdee55c353166a4446cde57eda0c4`.
