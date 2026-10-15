"""ComfyUI AMD MXFP8 Fix
======================

Workaround for MXFP8 (and other emulated quant) models that crash ComfyUI on
AMD / ROCm GPUs (RDNA3 confirmed: gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800
XT/7700 XT).

THE PROBLEM
-----------
On a GPU that cannot compute MXFP8 natively, ComfyUI still loads the weights as
quantized tensors and leaves the dequantization to *every forward pass*:

    # comfy/ops.py -> get_disabled_quant_formats()
    if not comfy.model_management.supports_mxfp8_compute(device):
        disabled.add("mxfp8")

    # comfy/ops.py -> _load_quantized_module()
    if module.quant_format in disabled_formats:
        module._full_precision_mm = True          # "emulated" mode

...and then, per forward:

    # comfy/ops.py -> forward_comfy_cast_weights()
    if self._full_precision_mm and isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()              # <-- every single forward

For a DiT with a few hundred quantized linears this allocates hundreds of large
temporary tensors per sampling step (a single [8192, 2048] layer produces a
64 MB fp32 temp). Inside ComfyUI's VRAM manager, on ROCm, that reproducibly
corrupts memory and kills the process at a random sampling step:

    Fatal Python error: Segmentation fault
      comfy_kitchen/backends/eager/quantization.py:365 in dequantize_mxfp8
      comfy_kitchen/tensor/mxfp8.py:81 in dequantize
      comfy/ops.py:1403 in forward_comfy_cast_weights

Note the signal: the fault address equals the instruction pointer (a jump to an
invalid address = corruption), it appears at a *random* step, there is no
``amdgpu`` page fault in the kernel log, and the very same dequantize op never
crashes when run standalone outside ComfyUI. It is also not related to tensor
shape/padding: all MXFP8 layers are stored in clean multiples of 32.

THE WORKAROUND
--------------
Dequantize once, at load time, and keep the weight as a plain tensor. Then
``layout_type`` is None, ``_use_quantized`` is False and the forward becomes a
plain ``F.linear`` — no per-forward dequantize, no large temporaries, no crash.

    # what this node does, per quantized module, during _load_from_state_dict:
    module.weight            = nn.Parameter(weight.dequantize().to(compute_dtype))
    module.layout_type       = None
    module._full_precision_mm = False

TRADE-OFFS
----------
* The model is held at its compute dtype (bf16), so the VRAM saving that made
  the quantized file attractive is LOST. Example: Anima Turbo MXFP8 goes from
  2198 MiB (fp8, crashing) to 3988 MiB (bf16, working).
* In exchange it stops crashing, and sampling is usually *faster*, because the
  matmul runs in bf16 instead of going through an emulated dequant every step.
* If you want both stability AND the small footprint, prefer an INT8
  (``int8_tensorwise``) build of the same model when one exists: INT8 is native
  on RDNA3, so it is not affected by this bug at all.

This node is a workaround, not a fix. The real fix belongs in comfy-kitchen /
ComfyUI (make the emulated path safe, or bound the temporary allocations). If
you can reproduce this, please also report it upstream.

CONFIGURATION
-------------
COMFY_AMD_MXFP8_FIX=0|1
    Enable/disable the patch. Default: 1 (enabled).

COMFY_AMD_MXFP8_FIX_FORMATS=mxfp8
    Comma-separated quant formats to dequantize at load. Default: ``mxfp8``
    (the narrowest, safest target). Use ``*`` to apply to every format that is
    *emulated* on your device (see ``get_disabled_quant_formats``), e.g.
    ``*`` would also cover ``nvfp4`` / ``float8_*`` on RDNA3.
"""

import logging
import os

__version__ = "1.0.0"

logger = logging.getLogger("comfyui-amd-mxfp8-fix")

_ENV_ON = "COMFY_AMD_MXFP8_FIX"
_ENV_FORMATS = "COMFY_AMD_MXFP8_FIX_FORMATS"
_DEFAULT_FORMATS = "mxfp8"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_STATS = {"dequantized": 0, "failed": 0, "formats": {}}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


ENABLED = _env_flag(_ENV_ON, True)
_FORMATS_RAW = os.environ.get(_ENV_FORMATS, _DEFAULT_FORMATS).strip()


def _wanted_formats() -> set:
    if _FORMATS_RAW in ("*", "all"):
        return set()  # filled dynamically from the device's emulated formats
    return {f.strip() for f in _FORMATS_RAW.split(",") if f.strip()}


def _install() -> bool:
    try:
        import torch  # noqa: F401
        import comfy.ops
        from comfy.ops import QuantizedTensor
    except Exception as exc:
        logger.warning("[mxfp8-fix] comfy.ops unavailable (%s); node inactive.", exc)
        return False

    original = getattr(comfy.ops, "_load_quantized_module", None)
    if original is None:
        logger.warning(
            "[mxfp8-fix] comfy.ops._load_quantized_module not found — this ComfyUI "
            "version changed its quantization internals. Node inactive. Please "
            "open an issue with your ComfyUI version."
        )
        return False

    if getattr(original, "_mxfp8_fix_wrapped", False):
        return True

    wanted = _wanted_formats()

    def _is_target(module) -> bool:
        fmt = getattr(module, "quant_format", None)
        if not fmt:
            return False
        if wanted:
            return fmt in wanted
        # formats "*": only what this device cannot compute natively
        try:
            emulated = comfy.ops.get_disabled_quant_formats(
                module.factory_kwargs.get("device")
            )
        except Exception:
            return False
        return fmt in emulated

    def patched(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        original(
            module, super_load, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs, load_extra_params,
        )

        if not _is_target(module):
            return

        weight = getattr(module, "weight", None)
        if not isinstance(weight, QuantizedTensor):
            return

        try:
            full = weight.dequantize()
            import torch

            dtype = module.factory_kwargs.get("dtype") or torch.bfloat16
            full = full.to(dtype)

            module.weight = torch.nn.Parameter(full, requires_grad=False)
            # No layout -> _use_quantized False -> plain F.linear, no per-forward
            # dequantize.
            module.layout_type = None
            module._full_precision_mm = False
        except Exception as exc:
            _STATS["failed"] += 1
            logger.warning(
                "[mxfp8-fix] could not dequantize %s at load (%s): %s — keeping it "
                "quantized; the emulated forward path may still crash.",
                (prefix or "").rstrip("."), getattr(module, "quant_format", "?"), exc,
            )
            return

        _STATS["dequantized"] += 1
        fmt = getattr(module, "quant_format", "?")
        _STATS["formats"][fmt] = _STATS["formats"].get(fmt, 0) + 1
        n = _STATS["dequantized"]
        if n == 1 or n % 128 == 0:
            logger.info("[mxfp8-fix] dequantized %d layers at load...", n)

    patched._mxfp8_fix_wrapped = True
    comfy.ops._load_quantized_module = patched

    try:
        import comfy.model_management as mm

        device = mm.get_torch_device()
    except Exception:
        device = None
    try:
        emulated = sorted(comfy.ops.get_disabled_quant_formats(device))
    except Exception:
        emulated = []

    if wanted:
        logger.info(
            "[mxfp8-fix] active — dequantizing at load: %s (emulated on this "
            "device: %s)",
            ", ".join(sorted(wanted)), ", ".join(emulated) or "-",
        )
    else:
        logger.info(
            "[mxfp8-fix] active (formats '*') — dequantizing every emulated "
            "format at load: %s",
            ", ".join(emulated) or "-",
        )
    return True


if ENABLED:
    _install()
else:
    logger.info(
        "[mxfp8-fix] disabled. Set %s=1 to enable the MXFP8 workaround.", _ENV_ON
    )
