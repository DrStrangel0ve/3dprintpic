# ZeroGPU selection-state boundary fix

## Reproduction

The deployed hover map successfully highlighted objects, but clicking one
returned `404: Selection precompute session not found`. The failure occurred
immediately after a successful SAM 3 precompute, so the 20-minute TTL was not
involved.

## Root cause

`@spaces.GPU` executes the preparation callback in a forked ZeroGPU worker.
The initial implementation wrote packed masks into a Python global and returned
only its opaque ID. The ID crossed the worker boundary in `gr.State`; the global
dictionary did not. The CPU click callback therefore looked up an ID that had
never existed in its process.

## Fix

- Return a versioned, pickle-safe packed-mask payload through `gr.State`.
- Include only SAM 3 instances represented by the score-resolved hover map.
- Keep source RGB pixels out of session state.
- Compress each represented bit-packed mask independently and enforce hard
  source-pixel, instance-count, and compressed-byte bounds.
- Set Gradio's state TTL so payloads are actually evicted after 20 minutes.
- Validate schema, age, model ID, dimensions, packed shape, and label count
  before materializing a selection.
- Preserve the old process-local lookup only as a compatibility fallback.

## Validation

The focused regression pickles and unpickles the complete preparation state to
simulate the ZeroGPU worker/main-process boundary, makes the global lookup fail
if called, and verifies that the exact highlighted person mask is selected
without another model invocation. Additional contracts cover non-contiguous
instance remapping, the compressed-payload limit, and Gradio TTL configuration.
