# Z-Image face-local transfer smoke (2026-07-19)

## Decision

**Hold.** Face-local 512 px transfer is a substantial improvement over native
256 px inference, and the corrected float-depth path improves the previous
best geometry again. It still does not clear every locked alignment gate or
make the synthetic face sufficiently natural. No derived corpus was published
and production behavior is unchanged.

The bounded smoke used the privacy-safe train-only row
`mhr_training_identity_000__scene_00` (49 x 75 px visible face, yaw -40
degrees), seed `20260719`, Z-Image-Turbo, and the 2602 eight-step ControlNet
Union 2.1 checkpoint. All background pixels in the final candidate remain
bit-exact to the parent image.

## Research basis

The official ControlNet model card says the current controls were retrained
with multi-resolution control images from 512 to 1536 pixels. The native 256
smoke was therefore below that training range. DiffSynth also requires output
dimensions divisible by 16. The challenger uses a deterministic 143 x 143
source crop around the exact selection, scales it to 512 x 512, and maps it
back through the recorded bbox and resampling settings before the exact
selection composite.

- Controller: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- Pinned DiffSynth documentation: https://github.com/modelscope/DiffSynth-Studio/blob/fb337fbb90945ff829de69dbd44ded618f73e889/docs/en/Model_Details/Z-Image.md

## Results

| Variant | Face IoU | Boundary H95 | Face height | Landmark median / p95 | RGB RMS | Exact parts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Native 256, d=0.35 | 0.8472 | 3.0 px | 1.0426 | 2.503 / 3.230 px | 29.08 | 4/6 | Hold |
| Crop 512, d=0.35 | 0.9420 | 2.0 px | 0.9787 | 1.218 / 1.695 px | 10.24 | 6/6 | Hold |
| Crop 512, quantized-control, d=0.25, c=0.9 | 0.9610 | 1.0 px | 0.9787 | 0.669 / **1.107** px | 7.78 | 5/6 | Hold |
| Crop 512, float-depth, d=0.25, c=0.9 | **0.9713** | **1.0 px** | 0.9787 | **0.637** / 1.132 px | 7.75 | 5/6 | **Hold, best geometry** |

The corrected 0.25 candidate clears face boundary, center, landmark median/p95,
non-noop transfer, background exactness, and five exact-part absolute gates.
It still misses the 0.98 face-IoU gate, the minimum face-height threshold by
0.0013, the combined relative part gate, and the mouth absolute gate (IoU
0.1045 versus 0.12). These thresholds were not relaxed. Visual inspection also
shows that the face remains recognizably synthetic despite the non-noop RGB
change, so this result is not suitable for corpus expansion.

The first three measured runs resized the already quantized inverse-depth control.
Commit `62f10010d00256c639e43df1d5dd5fa0aab81869` corrects the crop path to
resize floating-point exact depth first and quantize once afterward. Its 16
focused tests passed in both the project and pinned Torch environments. The
corrected one-row confirmation is now measured at exact commit
`1b2de69eed0248e38cfa89ad24897ca307103194`.

The official pinned Union example uses control scale 0.7. A paired c=0.7 run
was launched with every other input fixed, but timed out after 3,604 seconds.
It emitted a checksum-clean preflight and input/control crops only; it produced
no generated image or `result.json`, so it is an invalid comparison and is not
ranked above. Spending another local hour on this scale knob is not justified.

## Runtime compatibility

The pinned provider exposed several Windows-specific failures before inference
was reproducible: Torch 2.11 `torch_cpu.dll` access violations, paging-file
errors from simultaneous safetensor mappings, stale CUDA-backed safetensor
storage, and unwrapped meta pad tokens. The final exact path:

1. Uses pinned Torch `2.7.1+cu128`.
2. Encodes text before loading the DiT/ControlNet/VAE stage.
3. Reads pinned F32/BF16 tensor payloads from validated safetensors byte
   offsets into owned storage, then streams layers to CUDA.
4. Materializes only the root `x_pad_token` and `cap_pad_token` parameters that
   the official VRAM module map leaves on `meta`.
5. Uses an 8 GiB layer budget for the 512 px crop.

Provider source and all model/controller files were checksum-clean for the
successful measured runs. The corrected d=0.25/c=0.9 run used exact commit
`1b2de69eed0248e38cfa89ad24897ca307103194`, reported 1,164.50 seconds of
module runtime, and used 13.463 GiB peak allocated VRAM under the official
CUDA-staged low-VRAM path.

## Storage-integrity recovery

Repeated whole-file reads had shown isolated, changing 8 MiB regions while
file size, timestamps, and file identity stayed fixed. Whole-file retries were
therefore unlikely to obtain a completely clean 4-10 GB pass. Commit
`1b2de69eed0248e38cfa89ad24897ca307103194` replaces that recovery boundary
with fresh-handle 8 MiB block quorums: two byte-identical reads are required,
at most five votes are allowed, and rejected aggregate reads are retried. The
assembled file must still match the immutable official SHA-256. Safetensor
header parsing, full-file manifests, and per-tensor digests use the same
accepted byte stream; every tensor payload is quorum-read again and must match
its manifest digest before Torch conversion. No quorum can authorize bytes on
its own.

The exact successful preflight is runnable and has SHA-256
`40dd10401a4bfabc49901efdb719a0185e790ba165ca37bcb064047ce548e165`.
The 6.71 GB controller required 1,602 block reads (exactly two votes per
range), one aggregate attempt, zero retries, and zero mismatches. Runtime
verified 31,499,843,390 manifest bytes plus 195,831,970,438 tensor-payload
bytes in the main stage, and 8,044,982,000 manifest bytes plus 8,044,936,192
payload bytes in the text stage. All manifest, payload, quorum, size-change,
and short-read failure counters are zero.

The adversarial suite covers corrupt-clean-clean recovery, five-way no-quorum
failure, two matching corrupt votes followed by aggregate recovery, stable
post-manifest corruption, short reads, fresh handles, bounded header ranges,
cross-block tensor offsets, wrong official pins, and stale reader upgrades.
It passes 33/33 in both project and pinned Torch environments; the surrounding
matrix passes 74 tests plus 8 subtests. Summarized earlier observations remain
in `storage_integrity_diagnostics.json`.

The next bounded image-detail lead is the same Union-2602 depth candidate at
1024 x 1024. It changes only the provider crop size and retains the exact depth
conditioning, seed, prompt, denoising strength, control scale, and eight-step
schedule. The pinned official low-VRAM example already uses 1024 x 1024:
https://github.com/modelscope/DiffSynth-Studio/blob/fb337fbb90945ff829de69dbd44ded618f73e889/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py.

If that remains soft, the second lead is the official Apache-2.0
`Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps` checkpoint. Its model card
describes a retrained super-resolution model for high-resolution detail:
https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1.
Tile is a mouth/detail challenger, not evidence of better geometry. The pinned
pipeline accepts one ControlNet, so Tile must be a separate low-denoise pass
after Union rather than a replacement for depth conditioning.

## Next action

Close local 512 px denoising/control-scale tuning: the corrected c=0.9 result
still looks synthetic, and the official c=0.7 comparison did not complete in a
one-hour bound. Do not run 1024 px Union on the 3080 Ti; the 512 px path already
peaks at 13.463 GiB and takes about twenty minutes after verified admission.
Use 1024 or the separately pinned Tile-2601 mouth/detail challenger only on a
genuine >=80 GB G4 runtime. Meanwhile, move the face-depth work upstream to a
camera-aligned expressive geometry provider or a materially expanded
privacy-safe supervised corpus. Preserve the exact background, six-part,
raw-depth, dark-skin, eyewear, cast-shadow, and 30 mm printability gates.
