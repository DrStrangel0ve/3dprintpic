# MediaPipe canonical face fixture

- Upstream: `google-ai-edge/mediapipe`
- Commit: `a908d668c730da128dfa8d9f6bd25d519d006692`
- Path: `mediapipe/modules/face_geometry/data/canonical_face_model.obj`
- SHA256: `8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618`
- License: Apache-2.0; a pinned copy is included as `LICENSE`
- Attribution: MediaPipe canonical face model, Copyright 2020 The MediaPipe
  Authors, licensed under Apache-2.0.

The enclosing upstream BUILD package declares notice licensing and exports this
OBJ as the canonical reference face. The asset has 468 ordered landmark
vertices and 898 triangles. The official coordinate unit is one centimeter.

This is an open facial surface used only as a deterministic rendering and depth
oracle. It is not represented as a watertight or printable full head.
