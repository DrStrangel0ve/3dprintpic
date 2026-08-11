# Hugging Face Space SAM 3 unlock smoke

Date: 2026-08-11

Space: `DrStrangel0ve/3dprintpic` on ZeroGPU

## Configuration

- Created a dedicated read-only Hugging Face access token for the Space owner.
- Stored it only as the private Space secret `HF_TOKEN`.
- Cleared the transient browser clipboard and did not write the token to the
  repository, logs, generated artifacts, or local configuration.
- Added `HF_XET_CACHE=/tmp/3dprintpic-space/huggingface-xet` because the first
  gated download attempted to create Xet logs below the read-only
  `/home/user/.cache/huggingface/xet` path.

## Live selection smoke

The smoke used the public Windows system wallpaper
`C:/Windows/Web/Wallpaper/ThemeC/img29.jpg`; no private user image was uploaded.

1. The first request authenticated to `facebook/sam3` and reached the pinned
   `3c879f39826c281e95690f02c7821c4de09afae7` checkpoint, but failed closed on
   the Xet cache permission error.
2. After moving only the Xet cache to writable temporary storage and restarting
   the Space, the 3.44 GB `model.safetensors` download completed, both SAM 3
   model heads loaded, and the UI returned `Selected object with SAM 3`.
3. The selected-object state enabled the Full Mesh action, confirming that the
   former missing-secret lock is removed.

## ZeroGPU duration finding

The original `@spaces.GPU(duration=240)` Full Mesh request was converted by the
live scheduler into a 360-second reservation. The anonymous smoke was rejected
with HTTP 429 before TripoSG code ran, and the same reservation cannot fit a
free account's five-minute daily quota. The Space now requests 150 seconds for
Full Mesh, leaving room for the preceding 45-second SAM 3 selection while still
covering the measured single-image TripoSG inference path.

## Regression coverage

`python -m unittest huggingface_space.tests.test_space_runtime huggingface_space.tests.test_space_ui`

Result: 12 tests passed. The suite now asserts both the writable Xet cache
contract and the 150-second Full Mesh duration.
