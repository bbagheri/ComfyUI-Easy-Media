from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("easy_media.h3_motion_context")
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
CONTEXT_SWAP_NOISE_ALPHA = 0.45
CONTEXT_SWAP_NOISE_ALPHA_END = 0.10
CONTEXT_SWAP_NOISE_RAMP_STEPS = 2
REENCODED_ANCHOR_FRAMES = 5


def _pixel_frames(latent_steps: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(latent_steps))


def _step_offsets(latent_steps: int) -> list[int]:
    offsets: list[int] = []
    frame = 0
    for index in range(latent_steps):
        offsets.append(frame)
        frame += FRAME_PER_TOKEN[index % 5]
    return offsets


def _streams_from_latent(latent: dict[str, Any]) -> list[torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("easy h3 motion context: expected an H3 AV latent")
    samples = latent["samples"]
    if isinstance(samples, (tuple, list)):
        parts = list(samples)
    elif isinstance(getattr(samples, "tensors", None), (tuple, list)):
        parts = list(samples.tensors)
    elif getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    else:
        raise ValueError(
            "easy h3 motion context: expected a nested video/audio latent, "
            f"got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("easy h3 motion context: AV latent contains no streams")
    return parts


def _official_nested_tensor(parts: tuple[torch.Tensor, ...]) -> Any:
    """Rebuild H3 AV data with ComfyUI's supported NestedTensor wrapper."""
    try:
        import comfy.nested_tensor
    except ImportError as error:
        raise RuntimeError(
            "easy h3 motion context: ComfyUI nested tensor support is unavailable"
        ) from error
    return comfy.nested_tensor.NestedTensor(parts)


def _noise_mask_streams(
    latent: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return optional video/audio masks, including legacy video-only masks."""
    mask = latent.get("noise_mask")
    if mask is None:
        return None, None
    if isinstance(mask, torch.Tensor):
        return mask, None
    if isinstance(mask, (tuple, list)):
        parts = tuple(mask)
    elif isinstance(getattr(mask, "tensors", None), (tuple, list)):
        parts = tuple(mask.tensors)
    elif getattr(mask, "is_nested", False) and hasattr(mask, "unbind"):
        parts = tuple(mask.unbind())
    else:
        raise ValueError(
            "easy h3 context: noise_mask must be a tensor or nested streams"
        )
    if not parts or len(parts) > 2 or not all(
        isinstance(part, torch.Tensor) for part in parts
    ):
        raise ValueError("easy h3 context: noise_mask contains invalid streams")
    return parts[0], parts[1] if len(parts) > 1 else None


def _merge_noise_mask(
    generated: torch.Tensor,
    existing: torch.Tensor | None,
    stream_name: str,
) -> torch.Tensor:
    """Preserve existing locks while adding a context release mask."""
    if existing is None:
        return generated
    existing = existing.to(device=generated.device, dtype=generated.dtype)
    try:
        return torch.minimum(generated, existing).contiguous()
    except RuntimeError as error:
        raise ValueError(
            f"easy h3 context: existing {stream_name} noise_mask shape "
            f"{tuple(existing.shape)} is incompatible with {tuple(generated.shape)}"
        ) from error


def _video_from_latent(latent: dict[str, Any]) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("easy h3 motion context: expected a video latent")
    samples = latent["samples"]
    # A directly VAE-encoded anchor is a normal ComfyUI video LATENT, while
    # sampled H3 data stores video and audio in a nested tensor.
    if isinstance(samples, torch.Tensor) and not getattr(samples, "is_nested", False):
        video = samples
    else:
        video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "easy h3 motion context: expected video latent [B,C,T,H,W], "
            f"got shape {tuple(video.shape)}"
        )
    return video


def _steps_for_frames(frame_count: int) -> int | None:
    steps = 0
    covered = 0
    while covered < frame_count:
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == frame_count else None


def _video_tail_from_latent(
    latent: dict[str, Any], frame_count: int
) -> tuple[list[torch.Tensor], list[int], int]:
    video, start, steps, covered = _video_tail_bounds(latent, frame_count)
    blocks = [
        video[:1, :, start + index : start + index + 1].clone()
        for index in range(steps)
    ]
    return blocks, _step_offsets(steps), covered


def _video_tail_bounds(
    latent: dict[str, Any], frame_count: int
) -> tuple[torch.Tensor, int, int, int]:
    """Validate an H3 tail and return its source tensor and temporal bounds."""
    video = _video_from_latent(latent)
    total_steps = int(video.shape[2])
    steps = _steps_for_frames(frame_count)
    if steps is None:
        raise ValueError(
            "easy h3 motion context: context length cannot be represented by "
            "whole H3 latent steps"
        )
    if steps > total_steps:
        raise ValueError(
            f"easy h3 motion context: requested {steps} latent steps, but the "
            f"context latent contains {total_steps}"
        )
    start = total_steps - steps
    if start % 5 != 0:
        raise RuntimeError(
            "easy h3 motion context: the context tail starts at an invalid "
            "H3 temporal cycle position"
        )
    covered = _pixel_frames(steps)
    return video, start, steps, covered


def _audio_tail_from_latent(
    latent: dict[str, Any], audio_frames: int
) -> tuple[torch.Tensor, int, float]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "easy h3 motion context: context_latent has no audio stream"
        )
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "easy h3 motion context: expected audio latent [B,C,2,T], "
            f"got shape {tuple(audio.shape)}"
        )
    total_steps = int(audio.shape[-1])
    video_frames = _pixel_frames(int(video.shape[2]))
    overhang = total_steps - FRAME_RESCALE * video_frames
    if not -0.5 < overhang < 0.5:
        LOGGER.warning(
            "Unexpected H3 audio grid (%d steps for %d frames); ignoring overhang",
            total_steps,
            video_frames,
        )
        overhang = 0.0
    requested_steps = int(round(audio_frames / FPS * AUDIO_HZ))
    if requested_steps > total_steps:
        LOGGER.warning(
            "Requested %d audio steps but only %d are available",
            requested_steps,
            total_steps,
        )
        requested_steps = total_steps
    if requested_steps < 1:
        raise ValueError("easy h3 motion context: audio context window is empty")
    return (
        audio[:1, ..., total_steps - requested_steps :].clone(),
        requested_steps,
        float(overhang),
    )


def _sanitize_context_video_latent(
    video: torch.Tensor,
    strength: float | None = None,
) -> torch.Tensor:
    """Damp high-frequency latent checkerboard grain, soft-clamp spikes, and anchor std."""
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 5
        or not torch.is_floating_point(video)
        or int(video.shape[3]) < 8
        or int(video.shape[4]) < 8
    ):
        return video

    if strength is None:
        try:
            strength = float(
                os.environ.get(
                    "EASYMEDIA_H3_CONTEXT_DEGRAIN_STRENGTH",
                    os.environ.get("EASYMEDIA_H3_LATENT_SANITIZE_STRENGTH", "0.65"),
                )
            )
        except (TypeError, ValueError):
            strength = 0.65
    scale = max(0.0, min(2.0, float(strength)))
    if scale <= 0.0:
        return video

    orig_dtype = video.dtype
    x = video.float()
    b, c, t, h, w = x.shape
    # 1. Per-channel soft-knee outlier clamp to stop variance spikes
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    std = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)
    z = (x - mean) / std
    knee = max(2.8, 3.8 - 0.5 * max(0.0, scale - 0.65))
    excess = (z.abs() - knee).clamp(min=0.0)
    z_clamped = z.sign() * (
        z.abs().clamp(max=knee) + torch.tanh(excess * 0.5) * 1.8
    )
    x = mean + z_clamped * std

    # 2. Global std inflation anchor (prevents 0.97 -> 1.12 std creep across 10+ segments)
    global_std = x.std().clamp(min=1e-5)
    target_std = 0.985
    if global_std > target_std:
        std_pull = min(0.85, 0.45 * (scale / 0.65))
        anchored_std = target_std + (global_std - target_std) * (1.0 - std_pull)
        x = mean + (x - mean) * (anchored_std / global_std)
        std = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

    # 3. Edge-preserving spatial high-frequency damping across (H_lat, W_lat)
    alpha_lat = min(0.75, 0.25 * (scale / 0.65))
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=torch.float32)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    padded = F.pad(flat, (1, 1, 1, 1), mode="reflect")
    blurred = F.conv2d(padded, kernel_2d, groups=c)
    delta = flat - blurred
    tau_factor = 0.35 * max(0.85, scale / 0.65)
    tau_sq = (tau_factor * std.permute(0, 2, 1, 3, 4).expand(b, t, c, 1, 1).reshape(b * t, c, 1, 1)).square()
    weight = torch.exp(-delta.square() / (2.0 * tau_sq))
    cleaned = flat - (alpha_lat * weight) * delta
    return cleaned.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(dtype=orig_dtype).contiguous()


def trim_motion_context_latent(
    latent: dict[str, Any],
    context_length: int | str = "22",
) -> dict[str, Any]:
    """Copy only the H3 AV tail needed by the next context segment to CPU."""
    frame_count = int(context_length)
    video, start, steps, covered = _video_tail_bounds(latent, frame_count)
    audio, _, _ = _audio_tail_from_latent(latent, covered)
    video = _sanitize_context_video_latent(video[:1, :, start : start + steps].detach())
    streams = tuple(
        stream.detach().to(device="cpu", copy=True).contiguous()
        for stream in (video, audio)
    )
    output = {"samples": _official_nested_tensor(streams)}
    anchor_samples = latent.get("anchor_samples")
    if anchor_samples is not None:
        if not isinstance(anchor_samples, torch.Tensor) or anchor_samples.ndim != 5:
            raise ValueError(
                "easy h3 motion context: anchor_samples must be a video latent"
            )
        output["anchor_samples"] = (
            _sanitize_context_video_latent(anchor_samples.detach())
            .to(device="cpu", copy=True)
            .contiguous()
        )
    return output


def apply_context_swap_noise(
    latent: dict[str, Any],
    context_length: int | str = "22",
    seed: int = 0,
    alpha: float = CONTEXT_SWAP_NOISE_ALPHA,
    alpha_end: float = CONTEXT_SWAP_NOISE_ALPHA_END,
    ramp_steps: int = CONTEXT_SWAP_NOISE_RAMP_STEPS,
) -> dict[str, Any]:
    """Return a disposable context copy with tapered video-latent noise."""
    alpha = float(alpha)
    alpha_end = float(alpha_end)
    if not 0.0 <= alpha_end <= alpha <= 1.0:
        raise ValueError(
            "easy h3 context swap: expected 0 <= alpha_end <= alpha <= 1"
        )

    parts = _streams_from_latent(latent)
    source_video, start, steps, covered = _video_tail_bounds(
        latent,
        int(context_length),
    )
    ramp = min(max(1, int(ramp_steps)), steps)
    output_video = source_video.clone()
    tail = output_video[:, :, start : start + steps]
    scale = float(tail.detach().float().std().item())
    if scale != scale or scale in {float("inf"), float("-inf")} or scale <= 0.0:
        LOGGER.warning(
            "Context swap latent has a degenerate standard deviation; using unit noise"
        )
        scale = 1.0

    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) & 0xFFFFFFFFFFFFFFFF
    )
    schedule: list[float] = []
    for index in range(steps):
        from_end = steps - 1 - index
        amount = (
            alpha
            if from_end >= ramp
            else alpha + (alpha_end - alpha) * (ramp - from_end) / ramp
        )
        schedule.append(round(amount, 4))
        block = output_video[:, :, start + index]
        noise = torch.randn(
            tuple(block.shape),
            generator=generator,
            dtype=torch.float32,
        ).mul_(scale)
        noise = noise.to(device=block.device, dtype=block.dtype)
        output_video[:, :, start + index] = (
            block * (1.0 - amount) + noise * amount
        )

    if parts[0].ndim == 4:
        output_video = output_video[0]
    output_parts = list(parts)
    output_parts[0] = output_video
    output = latent.copy()
    output["samples"] = _official_nested_tensor(tuple(output_parts))
    LOGGER.info(
        "Context swap noise: video=%d steps/%d frames alpha=%s; audio untouched",
        steps,
        covered,
        schedule,
    )
    return output


def _resize_frames(
    images: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    try:
        import comfy.utils
    except ImportError as error:
        raise RuntimeError("ComfyUI image resize utilities are unavailable") from error
    samples = images[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples,
        width,
        height,
        "lanczos",
        "disabled",
    )
    return samples.movedim(1, -1)


def _encode_tail_audio(
    audio_vae: Any,
    audio: dict[str, Any],
    seconds: float,
) -> tuple[torch.Tensor, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise ValueError("easy h3 motion context: invalid context_audio")
    target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != target_rate:
        try:
            import torchaudio
        except ImportError as error:
            raise RuntimeError(
                "torchaudio is required to resample H3 context audio"
            ) from error
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_rate,
        )
    wanted = int(round(seconds * target_rate))
    available = int(waveform.shape[-1])
    if available > wanted:
        waveform = waveform[..., available - wanted :]
    encoded = audio_vae.encode(waveform[:1].movedim(1, -1))
    return encoded, int(encoded.shape[-1])


def _release_mask_inside_prefix(
    mask: torch.Tensor,
    prefix_steps: int,
    transition_steps: int,
) -> tuple[int, list[float]]:
    """Lock a copied prefix, then release it before generated latent begins."""
    prefix_steps = int(prefix_steps)
    ramp_steps = max(0, min(int(transition_steps), prefix_steps))
    locked_steps = prefix_steps - ramp_steps
    if locked_steps > 0:
        mask[..., :locked_steps] = 0.0
    ramp_values: list[float] = []
    if ramp_steps > 0:
        values = torch.arange(
            1,
            ramp_steps + 1,
            device=mask.device,
            dtype=mask.dtype,
        ) / float(ramp_steps + 1)
        shape = [1] * mask.ndim
        shape[-1] = ramp_steps
        mask[..., locked_steps:prefix_steps] = values.view(*shape)
        ramp_values = [round(float(value), 4) for value in values.detach().cpu()]
    return locked_steps, ramp_values


def _native_keyframe_stats(conditioning: Any) -> tuple[int, int]:
    """Count 0.4-style video/audio keyframes for compatibility checks."""
    max_video = 0
    max_audio = 0
    for _embedding, metadata in conditioning:
        video = 0
        audio = 0
        for keyframe in metadata.get("minimax_keyframes") or []:
            if keyframe.get("latent") is not None:
                video += 1
            if keyframe.get("audio_latent") is not None:
                audio += 1
        max_video = max(max_video, video)
        max_audio = max(max_audio, audio)
    return max_video, max_audio


def _hard_av_latent(
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    video_frames: int,
    video_transition_steps: int,
    audio_transition_steps: int,
) -> dict[str, Any]:
    """Copy previous video/audio tails into the current H3 sampling seed."""
    existing_video_mask, existing_audio_mask = _noise_mask_streams(latent)
    target_parts = _streams_from_latent(latent)
    if len(target_parts) < 2:
        raise ValueError("easy h3 hard context: target latent has no audio stream")
    target_video, target_audio = target_parts[:2]
    if target_video.ndim == 4:
        target_video = target_video.unsqueeze(0)
    if target_audio.ndim == 3:
        target_audio = target_audio.unsqueeze(0)
    if (
        target_video.ndim != 5
        or target_audio.ndim != 4
        or int(target_audio.shape[2]) != 2
    ):
        raise ValueError("easy h3 hard context: expected H3 video/audio latent streams")
    if int(target_video.shape[0]) != int(target_audio.shape[0]):
        raise ValueError(
            "easy h3 hard context: target video/audio batch sizes differ"
        )

    video_blocks, _, covered = _video_tail_from_latent(
        context_latent,
        int(video_frames),
    )
    copied_video = torch.cat(video_blocks, dim=2)
    video_steps = int(copied_video.shape[2])
    if covered != int(video_frames):
        raise RuntimeError("easy h3 hard context: video context span changed")
    if video_steps < 1 or video_steps >= int(target_video.shape[2]):
        raise ValueError(
            "easy h3 hard context: copied video prefix must be shorter than target"
        )
    if (
        copied_video.shape[0] != target_video.shape[0]
        or copied_video.shape[1] != target_video.shape[1]
        or copied_video.shape[3:] != target_video.shape[3:]
    ):
        raise ValueError(
            "easy h3 hard context: context and target video latent shapes differ"
        )
    output_video = target_video.clone()
    output_video[:, :, :video_steps] = copied_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    temporal_mask = video_mask.permute(0, 1, 3, 4, 2)
    video_locked, video_ramp = _release_mask_inside_prefix(
        temporal_mask,
        video_steps,
        video_transition_steps,
    )
    video_mask = temporal_mask.permute(0, 1, 4, 2, 3).contiguous()

    copied_audio, audio_steps, overhang = _audio_tail_from_latent(
        context_latent,
        int(video_frames),
    )
    if audio_steps < 1 or audio_steps >= int(target_audio.shape[-1]):
        raise ValueError(
            "easy h3 hard context: copied audio prefix must be shorter than target"
        )
    if copied_audio.shape[:3] != target_audio.shape[:3]:
        raise ValueError(
            "easy h3 hard context: context and target audio latent shapes differ"
        )
    output_audio = target_audio.clone()
    output_audio[..., :audio_steps] = copied_audio.to(
        device=output_audio.device,
        dtype=output_audio.dtype,
    )
    audio_mask = torch.ones_like(output_audio[:, :1], dtype=torch.float32)
    audio_locked, audio_ramp = _release_mask_inside_prefix(
        audio_mask,
        audio_steps,
        audio_transition_steps,
    )
    video_mask = _merge_noise_mask(
        video_mask,
        existing_video_mask,
        "video",
    )
    audio_mask = _merge_noise_mask(
        audio_mask,
        existing_audio_mask,
        "audio",
    )
    output = latent.copy()
    output["samples"] = _official_nested_tensor((output_video, output_audio))
    output["noise_mask"] = _official_nested_tensor((video_mask, audio_mask))
    LOGGER.info(
        "Hard AV context: video=%d steps/%d frames (lock=%d ramp=%s), "
        "audio=%d steps/%d frames (overhang=%.3f lock=%d ramp=%s)",
        video_steps,
        covered,
        video_locked,
        video_ramp or "off",
        audio_steps,
        video_frames,
        overhang,
        audio_locked,
        audio_ramp or "off",
    )
    return output


def _resolve_context_reset_denoise(reset_denoise: float | None = None) -> float:
    if reset_denoise is None:
        try:
            reset_denoise = float(
                os.environ.get("EASYMEDIA_H3_CONTEXT_RESET_DENOISE", "0.0")
            )
        except (TypeError, ValueError):
            reset_denoise = 0.0
    return max(0.0, min(1.0, float(reset_denoise)))


def _extract_ref_video_stats(
    conditioning: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract average per-channel mean/std from clean <Picture N> minimax_refs latents."""
    if not isinstance(conditioning, (list, tuple)):
        return None
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        meta = item[1]
        if not isinstance(meta, dict):
            continue
        for ref in meta.get("minimax_refs") or ():
            if not isinstance(ref, dict):
                continue
            lat = ref.get("latent")
            if (
                isinstance(lat, torch.Tensor)
                and lat.ndim in (4, 5)
                and int(lat.shape[1] if lat.ndim == 5 else lat.shape[0]) == 24
                and int(lat.shape[-2]) >= 8
                and int(lat.shape[-1]) >= 8
            ):
                x = lat.unsqueeze(0) if lat.ndim == 4 else lat[:1]
                x = x.detach().float()
                means.append(x.mean(dim=(2, 3, 4), keepdim=True))
                stds.append(x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5))
    if not means:
        return None
    ref_mean = torch.stack(means, dim=0).mean(dim=0)
    ref_std = torch.stack(stds, dim=0).mean(dim=0)
    return ref_mean, ref_std


def _reset_context_blocks_to_ref(
    blocks: list[torch.Tensor],
    conditioning: Any = None,
    reset_denoise: float | None = None,
) -> list[torch.Tensor]:
    """Anchor context blocks to clean reference stats and strip HF grain so minimax_refs supplies texture."""
    r = _resolve_context_reset_denoise(reset_denoise)
    if r <= 0.0 or not blocks:
        return blocks
    first = blocks[0]
    if (
        not isinstance(first, torch.Tensor)
        or first.ndim != 5
        or int(first.shape[3]) < 8
        or int(first.shape[4]) < 8
    ):
        return blocks

    orig_dtype = first.dtype
    cat = torch.cat(blocks, dim=2).float()
    b, c, t, h, w = cat.shape
    ctx_mean = cat.mean(dim=(2, 3, 4), keepdim=True)
    ctx_std = cat.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

    # 1. AdaIN per-channel moment anchor toward clean <Picture N> minimax_refs
    ref_stats = _extract_ref_video_stats(conditioning)
    if ref_stats is not None:
        ref_mean, ref_std = ref_stats
        ref_mean = ref_mean.to(device=cat.device, dtype=torch.float32)
        ref_std = ref_std.to(device=cat.device, dtype=torch.float32)
        std_blend = min(0.85, r * 1.4)
        mean_blend = min(0.35, r * 0.5)
        target_std = ctx_std + std_blend * (ref_std - ctx_std)
        target_mean = ctx_mean + mean_blend * (ref_mean - ctx_mean)
        cat = target_mean + (cat - ctx_mean) * (target_std / ctx_std)
        ctx_std = cat.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

    # 2. Pose-preserving high-frequency strip on minimax_keyframes
    flat = cat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    k1 = torch.tensor([0.25, 0.5, 0.25], device=cat.device, dtype=torch.float32)
    k2 = torch.outer(k1, k1).view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    low = F.conv2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), k2, groups=c)
    high = flat - low
    tau_sq = (
        0.75 * ctx_std.permute(0, 2, 1, 3, 4).expand(b, t, c, 1, 1).reshape(b * t, c, 1, 1)
    ).square()
    edge_protect = torch.exp(-high.square() / (2.0 * tau_sq))
    hf_damping = min(0.80, r * 1.25)
    cleaned = (flat - (hf_damping * edge_protect) * high).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    cleaned = cleaned.to(dtype=orig_dtype).contiguous()
    return [cleaned[:, :, idx : idx + 1].clone() for idx in range(t)]


def _reencoded_anchor_av_latent(
    latent: dict[str, Any],
    anchor_latent: dict[str, Any],
    context_frames: int,
    anchor_frames: int = REENCODED_ANCHOR_FRAMES,
    reset_denoise: float | None = None,
    conditioning: Any = None,
) -> dict[str, Any]:
    """Pin a short re-encoded video anchor at the end of a soft prefix, optionally releasing fine texture."""
    context_steps = _steps_for_frames(int(context_frames))
    anchor_video_steps = _steps_for_frames(int(anchor_frames))
    if context_steps is None or anchor_video_steps is None:
        raise ValueError(
            "easy h3 anchor context: context and anchor lengths must align "
            "to whole H3 latent steps"
        )
    anchor_start = context_steps - anchor_video_steps
    if anchor_start < 0 or anchor_start % len(FRAME_PER_TOKEN) != 0:
        raise ValueError(
            "easy h3 anchor context: anchor must begin on an H3 temporal cycle"
        )

    existing_video_mask, existing_audio_mask = _noise_mask_streams(latent)
    target_parts = _streams_from_latent(latent)
    if len(target_parts) < 2:
        raise ValueError("easy h3 anchor context: target latent has no audio stream")
    target_video, target_audio = target_parts[:2]
    if target_video.ndim == 4:
        target_video = target_video.unsqueeze(0)
    if target_audio.ndim == 3:
        target_audio = target_audio.unsqueeze(0)
    if target_video.ndim != 5 or target_audio.ndim != 4:
        raise ValueError("easy h3 anchor context: invalid target AV latent dimensions")
    if context_steps >= int(target_video.shape[2]):
        raise ValueError(
            "easy h3 anchor context: context prefix must be shorter than target"
        )

    blocks, _, covered = _video_tail_from_latent(anchor_latent, int(anchor_frames))
    r = _resolve_context_reset_denoise(reset_denoise)
    if r > 0.0 and int(target_video.shape[3]) >= 8 and int(target_video.shape[4]) >= 8:
        blocks = _reset_context_blocks_to_ref(blocks, conditioning=conditioning, reset_denoise=r)
    anchor_video = torch.cat(blocks, dim=2)
    if covered != int(anchor_frames):
        raise RuntimeError("easy h3 anchor context: video anchor span changed")
    if (
        anchor_video.shape[0] != target_video.shape[0]
        or anchor_video.shape[1] != target_video.shape[1]
        or anchor_video.shape[3:] != target_video.shape[3:]
    ):
        raise ValueError(
            "easy h3 anchor context: anchor and target video latent shapes differ"
        )
    output_video = target_video.clone()
    output_video[:, :, anchor_start:context_steps] = anchor_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    if r > 0.0 and int(target_video.shape[3]) >= 8 and int(target_video.shape[4]) >= 8:
        # Quantize fractional anchor release to 1/256 so MiniMaxH3._token_grid_masks
        # and KSamplerX0Inpaint share the exact per-row effective sigma = m * sigma_v.
        for offset in range(anchor_video_steps):
            frac = 0.65 + 0.35 * (float(offset) / max(1, anchor_video_steps - 1))
            m_val = math.ceil(min(1.0, r * frac) * 256.0) / 256.0
            video_mask[:, :, anchor_start + offset] = m_val
    else:
        video_mask[:, :, anchor_start:context_steps] = 0.0

    output_audio = target_audio
    audio_mask = torch.ones_like(output_audio[:, :1], dtype=torch.float32)

    output = latent.copy()
    output["samples"] = _official_nested_tensor((output_video, output_audio))
    output["noise_mask"] = _official_nested_tensor(
        (
            _merge_noise_mask(video_mask, existing_video_mask, "video"),
            _merge_noise_mask(audio_mask, existing_audio_mask, "audio"),
        )
    )
    LOGGER.info(
        "Re-encoded video anchor: soft context=%d frames, hard anchor=%d frames "
        "(video steps=%d:%d); audio remains generative",
        context_frames,
        anchor_frames,
        anchor_start,
        context_steps,
    )
    return output


def _video_anchor_from_context_latent(
    context_latent: dict[str, Any],
    anchor_frames: int = REENCODED_ANCHOR_FRAMES,
) -> dict[str, torch.Tensor]:
    """Use an embedded pixel anchor or the matching context-video tail."""
    anchor_samples = context_latent.get("anchor_samples")
    if anchor_samples is not None:
        if not isinstance(anchor_samples, torch.Tensor) or anchor_samples.ndim != 5:
            raise ValueError(
                "easy h3 anchor context: anchor_samples must be [B,C,T,H,W]"
            )
        return {"samples": anchor_samples}
    blocks, _, covered = _video_tail_from_latent(context_latent, int(anchor_frames))
    if covered != int(anchor_frames):
        raise RuntimeError("easy h3 anchor context: video anchor span changed")
    return {"samples": torch.cat(blocks, dim=2)}


def apply_reencoded_anchor_motion_context(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    context_length: int | str = "22",
    anchor_length: int | str = str(REENCODED_ANCHOR_FRAMES),
    reset_denoise: float | None = None,
) -> tuple[Any, int, dict[str, Any]]:
    """Use long native Motion Context with a short video anchor."""
    output, trim_frames = apply_motion_context(
        conditioning=conditioning,
        vae=vae,
        latent=latent,
        context_length=context_length,
        audio_context_length=0,
        context_latent=context_latent,
        reset_denoise=reset_denoise,
    )
    video_keyframes, audio_keyframes = _native_keyframe_stats(output)
    if video_keyframes < 1 or audio_keyframes < 1:
        raise RuntimeError(
            "easy h3 anchor context requires Motion Context 0.4.0+ native "
            "video/audio keyframes and ComfyUI 0.34.0+; got "
            f"video_keyframes={video_keyframes}, audio_keyframes={audio_keyframes}"
        )
    anchored = _reencoded_anchor_av_latent(
        latent,
        _video_anchor_from_context_latent(context_latent, int(anchor_length)),
        context_frames=int(trim_frames),
        anchor_frames=int(anchor_length),
        reset_denoise=reset_denoise,
        conditioning=conditioning,
    )
    return output, trim_frames, anchored


def apply_hard_motion_context(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    context_length: int | str = "22",
    video_transition_steps: int = 4,
    audio_transition_steps: int = 4,
) -> tuple[Any, int, dict[str, Any]]:
    """Apply normal H3 layout conditioning plus hard AV latent continuity."""
    output, trim_frames = apply_motion_context(
        conditioning=conditioning,
        vae=vae,
        latent=latent,
        context_length=context_length,
        audio_context_length=0,
        context_latent=context_latent,
    )
    return build_hard_motion_context(
        conditioning=output,
        trim_frames=trim_frames,
        latent=latent,
        context_latent=context_latent,
        video_transition_steps=video_transition_steps,
        audio_transition_steps=audio_transition_steps,
    )


def build_hard_motion_context(
    conditioning: Any,
    trim_frames: int,
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    video_transition_steps: int = 4,
    audio_transition_steps: int = 4,
) -> tuple[Any, int, dict[str, Any]]:
    """Validate native conditioning and add hard continuity to its target."""
    video_keyframes, audio_keyframes = _native_keyframe_stats(conditioning)
    if video_keyframes < 1 or audio_keyframes < 1:
        raise RuntimeError(
            "easy h3 hard context requires Motion Context 0.4.0+ native "
            "video/audio keyframes and ComfyUI 0.34.0+; got "
            f"video_keyframes={video_keyframes}, audio_keyframes={audio_keyframes}"
        )
    hard_latent = _hard_av_latent(
        latent,
        context_latent,
        int(trim_frames),
        video_transition_steps,
        audio_transition_steps,
    )
    LOGGER.info(
        "Hard AV kept native conditioning (%d video keyframes, %d audio "
        "keyframes); minimax_refs untouched",
        video_keyframes,
        audio_keyframes,
    )
    return conditioning, trim_frames, hard_latent


def apply_hires_continuity(
    current_hires_latent: dict[str, Any],
    previous_hires_latent: dict[str, Any],
    context_length: int | str = "22",
    video_transition_steps: int = 4,
) -> tuple[dict[str, Any], int]:
    """Build the masked high-resolution seed for an H3 second pass."""
    current_parts = _streams_from_latent(current_hires_latent)
    previous_parts = _streams_from_latent(previous_hires_latent)
    if len(current_parts) < 2 or not previous_parts:
        raise ValueError("easy h3 hires continuity: missing H3 AV latent streams")
    current_video, current_audio = current_parts[:2]
    previous_video = previous_parts[0]
    if current_video.ndim == 4:
        current_video = current_video.unsqueeze(0)
    if previous_video.ndim == 4:
        previous_video = previous_video.unsqueeze(0)
    if current_audio.ndim == 3:
        current_audio = current_audio.unsqueeze(0)
    if current_video.ndim != 5 or previous_video.ndim != 5 or current_audio.ndim != 4:
        raise ValueError("easy h3 hires continuity: invalid H3 latent dimensions")
    if (
        previous_video.shape[0] != current_video.shape[0]
        or previous_video.shape[1] != current_video.shape[1]
        or previous_video.shape[3:] != current_video.shape[3:]
    ):
        raise ValueError(
            "easy h3 hires continuity: previous and current video resolutions differ"
        )

    blocks, _, covered = _video_tail_from_latent(
        previous_hires_latent,
        int(context_length),
    )
    copied_video = torch.cat(blocks, dim=2)
    copied_steps = int(copied_video.shape[2])
    if copied_steps < 1 or copied_steps >= int(current_video.shape[2]):
        raise ValueError(
            "easy h3 hires continuity: copied prefix must be shorter than target"
        )
    output_video = current_video.clone()
    output_video[:, :, :copied_steps] = copied_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    temporal_mask = video_mask.permute(0, 1, 3, 4, 2)
    locked_steps, ramp_values = _release_mask_inside_prefix(
        temporal_mask,
        copied_steps,
        video_transition_steps,
    )
    video_mask = temporal_mask.permute(0, 1, 4, 2, 3).contiguous()
    audio_mask = torch.zeros_like(current_audio[:, :1], dtype=torch.float32)
    output = current_hires_latent.copy()
    if "noise_mask" in output:
        LOGGER.info("HiRes continuity is replacing the incoming noise_mask")
    output["samples"] = _official_nested_tensor((output_video, current_audio))
    output["noise_mask"] = _official_nested_tensor((video_mask, audio_mask))
    LOGGER.info(
        "HiRes continuity: video=%d steps/%d frames (lock=%d ramp=%s); "
        "second-pass audio is frozen",
        copied_steps,
        covered,
        locked_steps,
        ramp_values or "off",
    )
    return output, int(covered)


def apply_hires_anchor_continuity(
    current_hires_latent: dict[str, Any],
    previous_hires_latent: dict[str, Any],
    context_length: int | str = "22",
    anchor_length: int | str = str(REENCODED_ANCHOR_FRAMES),
    reset_denoise: float | None = None,
) -> tuple[dict[str, Any], int]:
    """Pin a re-encoded boundary anchor during ordinary hi-res refine."""
    output = _reencoded_anchor_av_latent(
        current_hires_latent,
        _video_anchor_from_context_latent(
            previous_hires_latent,
            int(anchor_length),
        ),
        context_frames=int(context_length),
        anchor_frames=int(anchor_length),
        reset_denoise=reset_denoise,
    )
    masks = _streams_from_latent({"samples": output["noise_mask"]})
    samples = _streams_from_latent(output)
    if len(masks) < 2 or len(samples) < 2:
        raise ValueError("easy h3 hires anchor: missing H3 AV streams")
    masks[1] = torch.zeros_like(samples[1][:, :1], dtype=torch.float32)
    output["noise_mask"] = _official_nested_tensor(tuple(masks))
    LOGGER.info("HiRes anchor continuity: second-pass audio is frozen")
    return output, int(context_length)


def apply_motion_context(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    context_length: int | str,
    audio_context_length: int = 24,
    context_frames: torch.Tensor | None = None,
    context_latent: dict[str, Any] | None = None,
    audio_vae: Any | None = None,
    context_audio: dict[str, Any] | None = None,
    reset_denoise: float | None = None,
) -> tuple[Any, int]:
    """Attach previous-clip video/audio tails to MiniMax H3 conditioning."""
    target_video = _video_from_latent(latent)
    latent_steps = int(target_video.shape[2])
    width = int(target_video.shape[4]) * 16
    height = int(target_video.shape[3]) * 16
    target_frames = _pixel_frames(latent_steps)

    if context_latent is not None:
        source_video = _video_from_latent(context_latent)
        source_width = int(source_video.shape[4]) * 16
        source_height = int(source_video.shape[3]) * 16
        if (source_width, source_height) != (width, height):
            raise ValueError(
                "easy h3 motion context: context latent resolution "
                f"{source_width}x{source_height} does not match target "
                f"{width}x{height}"
            )
        if int(source_video.shape[1]) != int(target_video.shape[1]):
            raise ValueError(
                "easy h3 motion context: context and target latent channels differ"
            )
        available_frames = _pixel_frames(int(source_video.shape[2]))
        source_kind = "latent"
    else:
        if context_frames is None:
            raise ValueError(
                "easy h3 motion context: connect context_latent or context_frames"
            )
        available_frames = int(context_frames.shape[0])
        source_kind = "pixels"

    requested_frames = int(context_length)
    pinned_frames = min(requested_frames, available_frames)
    if pinned_frames < 1:
        raise ValueError("easy h3 motion context: no frames are available to pin")
    snapped_frames = next(
        grid for grid in VIDEO_RUN_GRID if grid <= pinned_frames
    )
    if snapped_frames != pinned_frames:
        LOGGER.warning(
            "Context length %d is off the H3 VAE grid; using %d",
            pinned_frames,
            snapped_frames,
        )
    pinned_frames = snapped_frames
    if pinned_frames >= target_frames:
        raise ValueError(
            "easy h3 motion context: the pinned context must be shorter than "
            "the target clip"
        )

    if source_kind == "latent":
        blocks, offsets, span = _video_tail_from_latent(
            context_latent,
            pinned_frames,
        )
    else:
        assert context_frames is not None
        tail = _resize_frames(
            context_frames[available_frames - pinned_frames :],
            width,
            height,
        )
        encoded = vae.encode(tail)
        if getattr(encoded, "ndim", 0) != 5:
            raise ValueError(
                "easy h3 motion context: video VAE returned a non-H3 latent"
            )
        steps = int(encoded.shape[2])
        offsets = _step_offsets(steps)
        span = _pixel_frames(steps)
        if span != pinned_frames:
            raise RuntimeError(
                "easy h3 motion context: the video VAE temporal grid changed"
            )
        blocks = [encoded[:, :, index : index + 1] for index in range(steps)]

    blocks = _reset_context_blocks_to_ref(
        blocks,
        conditioning=conditioning,
        reset_denoise=reset_denoise,
    )

    keyframes = [
        {
            "resolved_frame_index": position,
            "latent": block,
        }
        for position, block in zip(offsets, blocks)
    ]

    audio_keyframe: dict[str, Any] | None = None
    audio_frames = 0
    audio_steps = 0
    if context_latent is not None or context_audio is not None:
        audio_frames = int(audio_context_length) or span
        if context_latent is not None:
            audio_latent, audio_steps, overhang = _audio_tail_from_latent(
                context_latent,
                audio_frames,
            )
        else:
            if audio_vae is None or context_audio is None:
                raise ValueError(
                    "easy h3 motion context: context_audio requires audio_vae"
                )
            audio_latent, audio_steps = _encode_tail_audio(
                audio_vae,
                context_audio,
                audio_frames / FPS,
            )
            overhang = 0.0
        end_frame = float(span) + overhang / FRAME_RESCALE
        end_frame = round(FRAME_RESCALE * end_frame) / FRAME_RESCALE
        audio_keyframe = {
            "resolved_frame_index": end_frame - audio_steps / FRAME_RESCALE,
            "audio_latent": audio_latent,
        }

    output = []
    dropped_positions: list[int | float] = []
    for embedding, metadata in conditioning:
        values = metadata.copy()
        previous_keyframes = values.get("minimax_keyframes") or []
        kept_keyframes = []
        for keyframe in previous_keyframes:
            position = keyframe.get("resolved_frame_index", 0)
            if position >= target_frames:
                raise ValueError(
                    "easy h3 motion context: conditioning keyframe at frame "
                    f"{position} exceeds the {target_frames}-frame target"
                )
            if position < span:
                dropped_positions.append(position)
                continue
            kept_keyframes.append(dict(keyframe))
        native_keyframes = keyframes + (
            [audio_keyframe] if audio_keyframe is not None else []
        )
        values["minimax_keyframes"] = kept_keyframes + native_keyframes
        output.append([embedding, values])

    if dropped_positions:
        LOGGER.warning(
            "Dropped existing keyframe anchors inside the pinned head: %s",
            sorted(set(dropped_positions)),
        )
    LOGGER.info(
        "Pinned %d H3 frames as %d blocks; trim=%d, audio=%d frames/%d steps",
        pinned_frames,
        len(blocks),
        span,
        audio_frames,
        audio_steps,
    )
    return output, span


def _encode_head_audio(
    audio_vae: Any,
    audio: dict[str, Any],
    seconds: float,
) -> tuple[torch.Tensor, int]:
    """Encode the opening ``seconds`` of an H3 audio dict."""
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise ValueError("easy h3 outgoing constraint: invalid outgoing_audio")
    target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != target_rate:
        try:
            import torchaudio
        except ImportError as error:
            raise RuntimeError(
                "torchaudio is required to resample H3 outgoing constraint audio"
            ) from error
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_rate,
        )
    wanted = max(1, int(round(seconds * target_rate)))
    available = int(waveform.shape[-1])
    if available > wanted:
        waveform = waveform[..., :wanted]
    elif available < wanted:
        waveform = F.pad(waveform, (0, wanted - available))
    encoded = audio_vae.encode(waveform[:1].movedim(1, -1))
    return encoded, int(encoded.shape[-1])


def apply_outgoing_tail_constraint(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    outgoing_images: torch.Tensor | None,
    outgoing_audio: dict[str, Any] | None = None,
    audio_vae: Any | None = None,
    trim_frames: int = 0,
    delivered_frames: int = 120,
    end_constraint_strength: float = 0.75,
    reset_denoise: float | None = None,
    is_second_pass: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Pin the opening frames of segment N+1 at the post-delivery overhang boundary (trim_frames + delivered_frames)."""
    s_strength = max(0.0, min(1.0, float(end_constraint_strength)))
    if (
        s_strength <= 0.0
        or outgoing_images is None
        or not isinstance(outgoing_images, torch.Tensor)
        or outgoing_images.ndim != 4
        or int(outgoing_images.shape[0]) < 1
    ):
        return conditioning, latent

    existing_video_mask, existing_audio_mask = _noise_mask_streams(latent)
    target_parts = _streams_from_latent(latent)
    if len(target_parts) < 2:
        return conditioning, latent
    target_video, target_audio = target_parts[:2]
    if target_video.ndim == 4:
        target_video = target_video.unsqueeze(0)
    if target_audio.ndim == 3:
        target_audio = target_audio.unsqueeze(0)
    if target_video.ndim != 5 or target_audio.ndim != 4:
        return conditioning, latent

    latent_steps = int(target_video.shape[2])
    height = int(target_video.shape[3]) * 16
    width = int(target_video.shape[4]) * 16
    if width <= 32 and height <= 32:
        return conditioning, latent

    target_frames = _pixel_frames(latent_steps)
    avail_frames = int(outgoing_images.shape[0])
    pinned_frames = 5 if avail_frames >= 5 else 1
    safe_trim = max(0, int(trim_frames))
    safe_delivered = max(1, int(delivered_frames))
    nominal_end_frame = safe_trim + safe_delivered
    max_start_frame = max(0, target_frames - pinned_frames)
    outgoing_frame_index = min(nominal_end_frame, max_start_frame)
    if outgoing_frame_index <= safe_trim:
        LOGGER.warning(
            "Skipping outgoing tail constraint: target_frames=%d cannot fit trim=%d + pinned=%d",
            target_frames,
            safe_trim,
            pinned_frames,
        )
        return conditioning, latent

    head_clip = _resize_frames(outgoing_images[:pinned_frames], width, height)
    r_denoise = _resolve_context_reset_denoise(reset_denoise)

    output_conditioning = conditioning
    if conditioning is not None:
        encoded_kf = vae.encode(head_clip)
        if getattr(encoded_kf, "ndim", 0) != 5:
            raise ValueError("easy h3 outgoing constraint: video VAE returned a non-H3 latent")
        encoded_kf = _sanitize_context_video_latent(encoded_kf.detach())
        kf_steps = int(encoded_kf.shape[2])
        kf_blocks = [encoded_kf[:, :, idx : idx + 1].clone() for idx in range(kf_steps)]
        if r_denoise > 0.0:
            kf_blocks = _reset_context_blocks_to_ref(
                kf_blocks,
                conditioning=conditioning,
                reset_denoise=r_denoise,
            )
        kf_offsets = [outgoing_frame_index + off for off in _step_offsets(kf_steps)]
        outgoing_keyframes: list[dict[str, Any]] = [
            {
                "resolved_frame_index": int(pos),
                "latent": blk,
            }
            for pos, blk in zip(kf_offsets, kf_blocks)
            if int(pos) < target_frames
        ]
        if not is_second_pass and outgoing_audio is not None and audio_vae is not None:
            try:
                head_audio_latent, _head_audio_steps = _encode_head_audio(
                    audio_vae,
                    outgoing_audio,
                    pinned_frames / FPS,
                )
                outgoing_keyframes.append(
                    {
                        "resolved_frame_index": float(outgoing_frame_index),
                        "audio_latent": head_audio_latent,
                    }
                )
            except Exception as exc:
                LOGGER.warning("Outgoing audio keyframe encoding skipped: %s", exc)

        new_cond = []
        for embedding, metadata in conditioning:
            values = metadata.copy()
            prev_kfs = values.get("minimax_keyframes") or []
            kept = [
                dict(kf)
                for kf in prev_kfs
                if not (
                    outgoing_frame_index
                    <= float(kf.get("resolved_frame_index", -1))
                    < outgoing_frame_index + pinned_frames
                )
            ]
            values["minimax_keyframes"] = kept + outgoing_keyframes
            new_cond.append([embedding, values])
        output_conditioning = new_cond

    step_offsets = _step_offsets(latent_steps)
    anchor_step = 0
    for s_idx, f_off in enumerate(step_offsets):
        if f_off <= outgoing_frame_index:
            anchor_step = s_idx
        else:
            break

    min_safe_step = (_steps_for_frames(safe_trim) or 0) + 1
    if anchor_step < min_safe_step or anchor_step >= latent_steps:
        return output_conditioning, latent

    r_phase = anchor_step % len(FRAME_PER_TOKEN)
    pad_before = _pixel_frames(r_phase)
    parts_to_cat: list[torch.Tensor] = []
    if pad_before > 0:
        parts_to_cat.append(head_clip[:1].expand(pad_before, *head_clip.shape[1:]))
    parts_to_cat.append(head_clip)
    current_len = pad_before + int(head_clip.shape[0])
    if current_len < 22:
        parts_to_cat.append(head_clip[-1:].expand(22 - current_len, *head_clip.shape[1:]))
    phase_buffer = torch.cat(parts_to_cat, dim=0)[:22].contiguous()
    encoded_phase = vae.encode(phase_buffer)
    encoded_phase = _sanitize_context_video_latent(encoded_phase.detach())

    want_tokens = 2 if pinned_frames >= 5 else 1
    num_stamp = min(
        want_tokens,
        latent_steps - anchor_step,
        int(encoded_phase.shape[2]) - r_phase,
    )
    if num_stamp < 1:
        return output_conditioning, latent

    stamp_blocks = [
        encoded_phase[:, :, r_phase + idx : r_phase + idx + 1].clone()
        for idx in range(num_stamp)
    ]
    if r_denoise > 0.0:
        stamp_blocks = _reset_context_blocks_to_ref(
            stamp_blocks,
            conditioning=output_conditioning,
            reset_denoise=r_denoise,
        )

    output_video = target_video.clone()
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    straddles_cut = step_offsets[anchor_step] < outgoing_frame_index
    m_overhang = math.ceil(max(r_denoise, 1.0 - s_strength) * 256.0) / 256.0
    m_straddle = math.ceil(max(0.40, 1.0 - 0.55 * s_strength) * 256.0) / 256.0

    for idx in range(num_stamp):
        step_pos = anchor_step + idx
        blk = stamp_blocks[idx].to(device=output_video.device, dtype=output_video.dtype)
        if idx == 0 and straddles_cut:
            if is_second_pass:
                output_video[:, :, step_pos : step_pos + 1] = (
                    0.5 * output_video[:, :, step_pos : step_pos + 1] + 0.5 * blk
                )
            else:
                output_video[:, :, step_pos : step_pos + 1] = blk
            video_mask[:, :, step_pos] = m_straddle
        else:
            output_video[:, :, step_pos : step_pos + 1] = blk
            video_mask[:, :, step_pos] = m_overhang

    audio_mask = (
        existing_audio_mask.clone()
        if existing_audio_mask is not None
        else torch.zeros_like(target_audio[:, :1], dtype=torch.float32)
        if is_second_pass
        else torch.ones_like(target_audio[:, :1], dtype=torch.float32)
    )
    output_latent = latent.copy()
    output_latent["samples"] = _official_nested_tensor((output_video, target_audio))
    output_latent["noise_mask"] = _official_nested_tensor(
        (
            _merge_noise_mask(video_mask, existing_video_mask, "video"),
            audio_mask,
        )
    )
    LOGGER.info(
        "Applied outgoing tail constraint (%s): frame=%d (trim=%d + delivered=%d), "
        "pinned_frames=%d, latent_steps=%d:%d, strength=%.2f",
        "pass2" if is_second_pass else "pass1",
        outgoing_frame_index,
        safe_trim,
        safe_delivered,
        pinned_frames,
        anchor_step,
        anchor_step + num_stamp,
        s_strength,
    )
    return output_conditioning, output_latent
