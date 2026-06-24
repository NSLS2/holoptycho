# ViT mosaic shear/skew — race condition investigation notes

Status: **unresolved root cause**, reliable workaround known. Paused to go back
to single-scan testing.

## Symptom
- The live ViT mosaic (`vit/mosaic`) **shears**: real sample features that should
  be horizontal (constant fast axis, spanning slow) appear **diagonal**.
- It **"starts good then skews"** — the shear is small early in the scan and grows
  over the scan.
- It is **non-deterministic**: the *same* config produced a skewed mosaic on one
  run and a correct one on another. → it is a **race condition**, not a config or
  data bug.

## What was ruled OUT (verified across several runs)
For a skewed run (`4790dab…`, scan 415933) and others:
- `positions_um` is **clean**: smooth, monotonic slow ramp, consistent fast sweep
  (~137 px / 20-frame step within a line, correct line wraps), **no NaN gaps**.
- **Ratio is 10:1**: positions count == frame count; recorded raw scan confirms
  100000 encoder samples / 10000 frames. `upsample=10` is correct.
- **No dropped frames** causing desync: `ImageBatchOp` stores the true Eiger
  `frame_id` (`preprocess.py:210` `indices_to_add[counter] = image_index`), not a
  sequential counter, so drops would leave holes, not a shear. No gaps seen.
- Channel mapping correct: PandA `/INENC2`=fast(x), `/INENC3`=slow(y); pipeline
  reads these (`ptycho_holo.py` `pos_x_channel`/`pos_y_channel`).
- Grid size / ranges do **not** cause the shear (only canvas size / clipping).
- Frame-index-vs-position map is a smooth gradient → positions place frames in
  correct spatial order.

## What it IS (best current understanding)
- A **race between the ViT branch and the position branch**. When the ViT branch
  runs far **ahead** of `PointProcessorOp` (the "pending buffer has N frames"
  warnings reached 600–700), the mosaic shears. When they stay in sync, it's fine.
- The shear geometry: **fast-placement gets a slow-proportional offset**
  (`placed_fast ≈ true_fast + k·slow`). Since `positions_um` is clean, the drift is
  in the **patch↔index pairing** (the patch tagged frame `i` effectively belongs at
  a frame index that drifts as the scan progresses), NOT in the position values.
- Most-suspect code paths (unconfirmed):
  - `PtychoViTInferenceOp` batch→engine-sub-batch chunking loop (pred/indices could
    get misaligned under load) — see `vit_inference.py` `_compute_inner` `ebs`/chunk.
  - `SaveViTResult` pending-frame merge (`_pending_frames` / `mosaic_canvas.partition_pending`)
    and/or early `_ensure_canvas` allocation (`vit_inference.py:539`, allocates as
    soon as ANY position is finite — but a wrong origin only *shifts*, can't *shear*,
    so this alone isn't the whole story).

## Workaround (reliable)
- Pass **`--frame-write-stride 1`** (or `20`) to `config-from-tiled`. The heavy
  per-frame dp/inference Tiled writes **throttle the ViT branch**, so it can't race
  ahead of positions → no shear. This is a *side-effect* fix, not a real one, and is
  WAN-write-heavy (stride 1 backpressures the inference writer — use 20 if so).

## The catch-22 blocking diagnosis
- To find the bug we need, per frame: patch content + its index + the position it was
  placed at. The skewed runs (light default capture, stride 1000) **didn't save the
  inference patches** (`dp_frames_written≈16`), so old skewed runs can't be diagnosed.
- But capturing patches (`--frame-write-stride`) **throttles ViT and removes the
  shear**. So no run has both "skewed" + "patches".

## Proposed next step to break the catch-22
- Add an **env-gated local-disk dump** in `SaveViTResult` (e.g.
  `HOLOPTYCHO_DEBUG_DUMP=/tmp/vitdump`) that appends `(index, position_used, patch)`
  per batch to a local file. Local disk = zero WAN cost = does NOT throttle ViT, so
  the shear still happens while we capture the ingredients. One un-throttled scan
  then gives everything to pinpoint the race.
- Alternatively/also: reproduce on **replay** run *without* `--frame-write-stride`
  (deterministic, same pipeline, no live timing) — confirm the shear reproduces
  offline, then bisect locally.

## Useful run_uids
- `4790dabca3b245a3ac2a534aafa95dee` — SKEWED, no stride, clean positions, no patches.
- `af1161ce40564384b9a62e49dccaa47b` — CORRECT, `--frame-write-stride 20`, has
  inference patches (rows 0–~650 non-zero before WAN backpressure dropped the rest).
- Scan 415933 raw uid (geometry): `d807f16e-f197-4394-86d1-4a6396822c61`
  (NOTE: that recorded raw scan is 100×100=10000 frames; the live scans tagged with
  it were larger — geometry overridden via `--x-num/--y-num/--x-range/--y-range`).

## Repro command (live, un-throttled → should skew)
```
hp restart "$(pixi run -e client config-from-tiled --uid <UID> --mode vit \
  --x-num 300 --y-num 300 --x-range 28 --y-range 21 \
  --overshoot-factor 1.4 --allow-mid-scan-join --mosaic-slow-gate)"
```
Add `--frame-write-stride 20` to make it correct (workaround / throttle).
