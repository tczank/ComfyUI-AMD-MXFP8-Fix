# ComfyUI AMD MXFP8 Fix

A small ComfyUI custom node that stops **MXFP8** diffusion models from
**segfaulting ComfyUI on AMD/ROCm GPUs** (confirmed on RDNA3: RX 7900 XTX/XT,
gfx1100/gfx1101).

It is a **workaround**, not a real fix: it trades VRAM for stability by
dequantizing the MXFP8 weights **once, at load**, instead of on every forward
pass.

---

## Symptoms

You load an MXFP8 model (a CivitAI "MXFP8" / "FP8 for AMD" build, or anything
whose metadata says `"format": "mxfp8"`) and ComfyUI dies during sampling:

```
Fatal Python error: Segmentation fault

  comfy_kitchen/backends/eager/quantization.py:365 in dequantize_mxfp8
  comfy_kitchen/tensor/mxfp8.py:81 in dequantize
  comfy/ops.py:1403 in forward_comfy_cast_weights
```

Details that match this bug:

- It crashes at a **random** step of the sampler (step 4, step 6, ...), never at
  a fixed point.
- The fault address equals the instruction pointer (a jump to an invalid
  address → memory corruption), it is **not** a GPU page fault.
- Nothing about `amdgpu` appears in the kernel log.
- The exact same `dequantize_mxfp8` call **never** crashes when you run it by
  hand on the same tensors, outside ComfyUI.
- Other models (bf16, fp8, INT8) work fine — only the emulated MXFP8 path dies.

Your startup log will show MXFP8 as **emulated**:

```
[INFO] Native ops: int8_tensorwise, asym_w4a8_int8, convrot_w4a4 ,
       emulated ops: mxfp8, float8_e4m3fn, float8_e5m2, nvfp4
```

## Why it happens

There is no MXFP8 compute on RDNA3, so `get_disabled_quant_formats()` puts
`mxfp8` in the "disabled" set and `_load_quantized_module()` flips the layer to
`_full_precision_mm = True` (emulated). In that mode, `comfy/ops.py` calls
`weight.dequantize()` **on every forward**.

For a few hundred quantized linear layers per sampling step, that is a few
hundred large temporary tensors per step (one `[8192, 2048]` layer alone makes a
64 MB fp32 temp). Inside ComfyUI's VRAM manager on ROCm, that reproducibly
corrupts memory and the process segfaults.

## What this node does

During `_load_from_state_dict`, for each module whose `quant_format` is in the
target set, it replaces the quantized weight with a plain tensor:

```python
module.weight             = nn.Parameter(weight.dequantize().to(compute_dtype))
module.layout_type        = None      # -> _use_quantized == False
module._full_precision_mm = False     # -> no per-forward dequantize
```

The forward then becomes a normal `F.linear`. No per-forward dequantize, no
giant temporaries, no crash.

No ComfyUI file is edited — it monkeypatches `comfy.ops._load_quantized_module`
at import time, and does nothing at all if that function is missing.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tczank/ComfyUI-AMD-MXFP8-Fix
# restart ComfyUI
```

On startup you should see:

```
[INFO] [mxfp8-fix] active — dequantizing at load: mxfp8 (emulated on this device: float8_e4m3fn, float8_e5m2, mxfp8, nvfp4)
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `COMFY_AMD_MXFP8_FIX` | `1` | `0` disables the node entirely. |
| `COMFY_AMD_MXFP8_FIX_FORMATS` | `mxfp8` | Comma-separated formats to dequantize at load. `*` = every format that is *emulated* on your device. |

Only `mxfp8` is targeted by default, so nothing else changes. If other emulated
formats also crash for you, use `COMFY_AMD_MXFP8_FIX_FORMATS=*`.

## Trade-offs — read this

| | quantized (MXFP8) | with this node |
| --- | --- | --- |
| works on RDNA3 | ❌ segfault | ✅ |
| VRAM (example: Anima Turbo, 2B) | 2198 MiB | **3988 MiB** |
| sampling speed | — | usually **faster** (bf16 matmul) |

**You lose the VRAM saving** that made the MXFP8 file attractive. If you want
stability *and* a small footprint, use an **INT8** (`int8_tensorwise`) build of
the same model instead — INT8 is native on RDNA3 and is not affected by this
bug. Verified: an identical Anima Turbo INT8 build loads at 2145 MiB and
produces visually the same image, with no patch needed.

## Compatibility notes

- Tested against ComfyUI `0.37.0`, comfy-kitchen `0.2.35`,
  `torch 2.12.0+rocm7.2`, ROCm `7.2.4`, RX 7900 XTX (gfx1100).
- It hooks a private function (`comfy.ops._load_quantized_module`). If a future
  ComfyUI changes that internal, the node **logs a warning and disables
  itself** instead of breaking anything.
- Applies to diffusion models loaded through the mixed-precision ops path
  (i.e. models that carry quantization metadata). Plain bf16/fp16 models are
  never touched.

## This is a workaround — please report the crash upstream

The real fix has to happen in comfy-kitchen / ComfyUI: bound the temporary
allocations made by the emulated path. If you can reproduce the segfault,
please report it with your `emulated ops` line, GPU arch, ROCm/torch versions
and a minimal workflow:

- https://github.com/Comfy-Org/ComfyUI/issues
- https://github.com/Comfy-Org/comfy-kitchen/issues

## License

MIT — see [LICENSE](LICENSE).
