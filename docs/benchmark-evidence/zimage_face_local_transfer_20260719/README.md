# Z-Image face-local transfer smoke (2026-07-19)

## Decision

**Hold.** Face-local 512 px transfer is a substantial improvement over native
256 px inference, but the best measured candidate does not clear every locked
alignment gate. No derived corpus was published and production behavior is
unchanged.

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
back through the recorded inverse affine before the exact selection composite.

- Controller: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- Pinned DiffSynth documentation: https://github.com/modelscope/DiffSynth-Studio/blob/fb337fbb90945ff829de69dbd44ded618f73e889/docs/en/Model_Details/Z-Image.md

## Results

| Variant | Face IoU | Boundary H95 | Face height | Landmark median / p95 | RGB RMS | Exact parts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Native 256, d=0.35 | 0.8472 | 3.0 px | 1.0426 | 2.503 / 3.230 px | 29.08 | 4/6 | Hold |
| Crop 512, d=0.35 | 0.9420 | 2.0 px | 0.9787 | 1.218 / 1.695 px | 10.24 | 6/6 | Hold |
| Crop 512, d=0.25 | **0.9610** | **1.0 px** | 0.9787 | **0.669 / 1.107 px** | 7.78 | 5/6 | **Hold, best** |

The 0.25 candidate clears face boundary, center, landmark median/p95,
non-noop transfer, background exactness, and five exact-part absolute gates.
It still misses the 0.98 face-IoU gate, the minimum face-height threshold by
0.0013, the combined relative part gate, and the mouth absolute gate. These
thresholds were not relaxed.

These three measured runs resized the already quantized inverse-depth control.
Commit `62f10010d00256c639e43df1d5dd5fa0aab81869` corrects the crop path to
resize floating-point exact depth first and quantize once afterward. Its 16
focused tests pass in both the project and pinned Torch environments. The
corrected one-row GPU confirmation remains pending because the local GPU became
occupied by an unrelated interactive application after the fix; no competing
run was launched.

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

Provider source and all model/controller files remained checksum-clean. The
successful d=0.25 run used exact commit
`770fb48a0995edd14310a02fc7f7388ff11c2061` and completed in 1349.51 seconds.

## Next action

First rerun only the d=0.25, 512 px candidate from exact commit `62f1001` when
the 3080 Ti is uncontested. If it remains hold, close additional
denoising-strength tuning on this provider. The next face geometry lane should
improve the mouth/silhouette upstream or use the official tile/super-resolution
control in a separately pinned one-row smoke. It must retain the same exact
background, six-part, raw-depth, and 30 mm printability gates.
