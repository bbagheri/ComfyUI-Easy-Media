from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


H3_VAE_FRAME_CHUNK = 17
H3_VAE_FRAME_REMAINDER = 5
H3_VAE_TOKENS_PER_CHUNK = 5
H3_VAE_FINAL_TOKEN_DROP = 3
H3_FRAMES_PER_TOKEN = (1, 4, 4, 4, 4)
H3_DEFAULT_CONTEXT_DEGRAIN_STRENGTH = 0.65
H3_DEFAULT_CONTEXT_DEGRAIN_THRESHOLD = 0.025


def _degrain_context_suffix(
    images: torch.Tensor,
    strength: float | None = None,
    threshold: float | None = None,
) -> torch.Tensor:
    """Apply an edge-preserving bilateral micro-denoiser to context handoff frames.

    Damps low-amplitude 1-2px digital solver grain (|delta| < threshold) before
    VAEEncode builds the next segment's prefix latent, while leaving structural
    edges (eyes, lips, hair strands, clothing seams where |delta| >> threshold)
    100% sharp.
    """
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or not torch.is_floating_point(images)
        or int(images.shape[1]) < 8
        or int(images.shape[2]) < 8
    ):
        return images

    if strength is None:
        try:
            strength = float(
                os.environ.get(
                    "EASYMEDIA_H3_CONTEXT_DEGRAIN_STRENGTH",
                    str(H3_DEFAULT_CONTEXT_DEGRAIN_STRENGTH),
                )
            )
        except (TypeError, ValueError):
            strength = H3_DEFAULT_CONTEXT_DEGRAIN_STRENGTH
    if threshold is None:
        try:
            threshold = float(
                os.environ.get(
                    "EASYMEDIA_H3_CONTEXT_DEGRAIN_THRESHOLD",
                    str(H3_DEFAULT_CONTEXT_DEGRAIN_THRESHOLD),
                )
            )
        except (TypeError, ValueError):
            threshold = H3_DEFAULT_CONTEXT_DEGRAIN_THRESHOLD

    alpha = max(0.0, min(1.0, float(strength)))
    tau = float(threshold)
    if alpha <= 0.0 or tau <= 0.0:
        return images

    orig_dtype = images.dtype
    # [T, H, W, C] -> [T, C, H, W]
    x = images.permute(0, 3, 1, 2).float()
    channels = int(x.shape[1])
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=torch.float32)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    blurred = F.conv2d(padded, kernel_2d, groups=channels)
    delta = x - blurred
    rgb_channels = min(3, channels)
    diff_sq = delta[:, :rgb_channels].square().mean(dim=1, keepdim=True)
    weight = torch.exp(-diff_sq / (2.0 * tau * tau))
    cleaned = x - (alpha * weight) * delta
    return cleaned.permute(0, 2, 3, 1).to(dtype=orig_dtype).contiguous()


def selflift_transition_step(step_count: int, ratio: float) -> int:
    """Round a low-resolution NFE ratio to a valid SelfLift boundary."""
    if step_count < 2:
        raise ValueError("SelfLift requires at least two denoiser evaluations")
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("SelfLift transition_ratio must be between 0 and 1")
    nearest = int(math.floor(step_count * ratio + 0.5))
    return max(1, min(step_count - 1, nearest))


def h3_phase_aligned_context_start(
    total_frames: int,
    context_frames: int = 22,
) -> int:
    """Return the first frame of the smallest phase-aligned H3 VAE suffix."""
    frame_count = int(total_frames)
    requested_frames = int(context_frames)
    if frame_count < 1:
        raise ValueError("total_frames must be positive")
    if requested_frames < 1 or requested_frames > frame_count:
        raise ValueError(
            "context_frames must be positive and no greater than total_frames"
        )

    latent_steps = 0
    covered_frames = 0
    while covered_frames < requested_frames:
        covered_frames += H3_FRAMES_PER_TOKEN[
            latent_steps % len(H3_FRAMES_PER_TOKEN)
        ]
        latent_steps += 1
    if covered_frames != requested_frames:
        raise ValueError(
            "context_frames must align to the H3 temporal latent grid"
        )

    required_chunks = (
        latent_steps
        + H3_VAE_FINAL_TOKEN_DROP
        + H3_VAE_TOKENS_PER_CHUNK
        - 1
    ) // H3_VAE_TOKENS_PER_CHUNK
    available_chunks = (
        frame_count + H3_VAE_FRAME_CHUNK - 1
    ) // H3_VAE_FRAME_CHUNK
    first_chunk = max(0, available_chunks - required_chunks)
    return first_chunk * H3_VAE_FRAME_CHUNK


def h3_phase_aligned_video_suffix(
    images: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    """Pad to the H3 frame grid and return an encoder-phase-aligned suffix."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must have IMAGE shape [B, H, W, C]")
    frame_count = int(images.shape[0])
    if frame_count < 1:
        raise ValueError("images must contain at least one frame")

    aligned_frame_count = H3_VAE_FRAME_CHUNK * max(
        0,
        math.ceil((frame_count - H3_VAE_FRAME_REMAINDER) / H3_VAE_FRAME_CHUNK),
    ) + H3_VAE_FRAME_REMAINDER
    if aligned_frame_count > frame_count:
        padding = images[-1:].expand(
            aligned_frame_count - frame_count,
            *images.shape[1:],
        )
        images = torch.cat((images, padding), dim=0)
    start = h3_phase_aligned_context_start(
        aligned_frame_count,
        int(context_frames),
    )
    suffix = images[start:].contiguous()
    return _degrain_context_suffix(suffix)


def remove_output_files_by_prefix(
    output_directory: str | Path,
    filename_path: str,
) -> int:
    """Remove files whose output-relative path starts with ``filename_path``."""
    normalized = filename_path.strip().replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "filename_path must be a non-empty path prefix inside the output directory"
        )

    output_root = Path(output_directory).resolve()
    search_directory = output_root.joinpath(*parts[:-1]).resolve()
    try:
        search_directory.relative_to(output_root)
    except ValueError as error:
        raise ValueError("filename_path must stay inside the output directory") from error

    if not search_directory.exists():
        return 0
    if not search_directory.is_dir():
        raise ValueError(
            f"filename_path parent is not a directory: {search_directory}"
        )

    filename_prefix = parts[-1]
    deleted_count = 0
    try:
        candidates = list(search_directory.iterdir())
    except OSError as error:
        raise RuntimeError(
            f"Unable to inspect latent save directory: {search_directory}"
        ) from error

    for candidate in candidates:
        if not candidate.name.startswith(filename_prefix) or not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(f"Unable to remove latent save: {candidate}") from error
        deleted_count += 1
    return deleted_count


def flatten_media_inputs(value: Any) -> list[Any]:
    """Flatten nested input-list values while preserving media objects."""
    if value is None:
        return []
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_media_inputs(item))
        return flattened
    return [value]


def expand_image_inputs(value: Any) -> list[torch.Tensor]:
    """Expand nested IMAGE lists and batches into single-image tensors."""
    images: list[torch.Tensor] = []
    for batch in flatten_media_inputs(value):
        if not isinstance(batch, torch.Tensor):
            raise TypeError("images must contain only torch.Tensor values")
        if batch.ndim != 4 or batch.shape[0] < 1:
            raise ValueError(
                "Each image input must have shape [B, H, W, C] with B >= 1"
            )
        images.extend(batch.split(1, dim=0))
    return images
