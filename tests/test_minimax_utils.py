import importlib.util
from pathlib import Path

import pytest
import torch


def _load_minimax_utils():
    path = Path(__file__).parents[1] / "utils" / "minimax.py"
    try:
        spec = importlib.util.spec_from_file_location("minimax_utils_under_test", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, ImportError):
        return None


def test_expand_image_inputs_flattens_batches_and_nested_lists():
    module = _load_minimax_utils()
    assert (
        module is not None
    ), "MiniMax media normalization utilities must be importable"
    first_batch = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1)
    second_batch = torch.tensor([[[[2.0]]]])

    images = module.expand_image_inputs([[first_batch], second_batch])

    assert [image.item() for image in images] == [0.0, 1.0, 2.0]
    assert all(image.shape == (1, 1, 1, 1) for image in images)


@pytest.mark.parametrize(
    ("total_frames", "context_frames", "expected_start"),
    [
        (22, 22, 0),
        (120, 22, 102),
        (124, 22, 102),
        (240, 22, 221),
        (120, 39, 85),
    ],
)
def test_h3_phase_aligned_context_start_preserves_encoder_chunk_phase(
    total_frames,
    context_frames,
    expected_start,
):
    module = _load_minimax_utils()

    assert module.h3_phase_aligned_context_start(
        total_frames,
        context_frames,
    ) == expected_start


@pytest.mark.parametrize("total_frames", [22, 120, 124, 240])
def test_h3_phase_aligned_suffix_matches_full_encode_tail(total_frames):
    module = _load_minimax_utils()

    def encode_chunk_signatures(first_frame, frame_count):
        signatures = []
        chunk_size = module.H3_VAE_FRAME_CHUNK
        for offset in range(0, frame_count, chunk_size):
            chunk_start = first_frame + offset
            chunk_end = min(first_frame + frame_count, chunk_start + chunk_size)
            chunk = tuple(range(chunk_start, chunk_end))
            padded_chunk = chunk + (chunk[-1],) * (chunk_size - len(chunk))
            signatures.extend(
                (padded_chunk, token_index)
                for token_index in range(module.H3_VAE_TOKENS_PER_CHUNK)
            )
        return signatures[:-module.H3_VAE_FINAL_TOKEN_DROP]

    start = module.h3_phase_aligned_context_start(total_frames, 22)
    full = encode_chunk_signatures(0, total_frames)
    suffix = encode_chunk_signatures(start, total_frames - start)

    assert suffix == full[-7:]


def test_h3_phase_aligned_video_suffix_restores_five_frame_anchor_behavior():
    module = _load_minimax_utils()
    images = torch.arange(120, dtype=torch.float32).reshape(120, 1, 1, 1)

    suffix = module.h3_phase_aligned_video_suffix(images, 5)

    assert suffix[:, 0, 0, 0].tolist() == [119.0] * 5


def test_h3_phase_aligned_video_suffix_keeps_grid_aligned_real_frames():
    module = _load_minimax_utils()
    images = torch.arange(124, dtype=torch.float32).reshape(124, 1, 1, 1)

    suffix = module.h3_phase_aligned_video_suffix(images, 5)

    assert suffix[:, 0, 0, 0].tolist() == [119.0, 120.0, 121.0, 122.0, 123.0]


def test_five_frame_anchor_matches_last_chunk_of_22_frame_context():
    module = _load_minimax_utils()
    images = torch.arange(120, dtype=torch.float32).reshape(120, 1, 1, 1)

    anchor = module.h3_phase_aligned_video_suffix(images, 5)
    context = module.h3_phase_aligned_video_suffix(images, 22)

    assert torch.equal(anchor, context[-5:])


@pytest.mark.parametrize("context_frames", [0, 6, 121])
def test_h3_phase_aligned_context_start_rejects_invalid_context(context_frames):
    module = _load_minimax_utils()

    with pytest.raises(ValueError):
        module.h3_phase_aligned_context_start(120, context_frames)


@pytest.mark.parametrize(
    ("step_count", "ratio", "expected"),
    [(8, 0.6, 5), (10, 0.6, 6), (12, 0.6, 7), (2, 0.95, 1)],
)
def test_selflift_transition_step_uses_nfe_ratio(step_count, ratio, expected):
    module = _load_minimax_utils()

    assert module.selflift_transition_step(step_count, ratio) == expected


@pytest.mark.parametrize("step_count,ratio", [(1, 0.6), (8, 0.0), (8, 1.0)])
def test_selflift_transition_step_rejects_invalid_boundaries(step_count, ratio):
    module = _load_minimax_utils()

    with pytest.raises(ValueError):
        module.selflift_transition_step(step_count, ratio)


def test_degrain_context_suffix_damps_micro_grain_and_preserves_edges():
    module = _load_minimax_utils()
    torch.manual_seed(42)
    base = torch.full((5, 32, 32, 3), 0.5, dtype=torch.float32)
    # Add low-amplitude high-frequency checkerboard grain
    checker = (
        (torch.arange(32).view(-1, 1) + torch.arange(32).view(1, -1)) % 2 * 2 - 1
    ).float() * 0.012
    noisy = base + checker.view(1, 32, 32, 1)
    cleaned = module._degrain_context_suffix(noisy)
    assert (cleaned - base).pow(2).mean() < (noisy - base).pow(2).mean() * 0.35

    # Verify sharp high-contrast step edge is preserved (>99%)
    step_img = torch.zeros((5, 32, 32, 3), dtype=torch.float32)
    step_img[:, :, 16:, :] = 0.85
    cleaned_edge = module._degrain_context_suffix(step_img)
    edge_diff = (cleaned_edge[:, :, 16, :] - cleaned_edge[:, :, 15, :]).mean().item()
    assert edge_diff > 0.84


def test_degrain_context_suffix_structure_preserving_gate_keeps_fluid_filaments():
    module = _load_minimax_utils()
    base = torch.full((5, 32, 32, 3), 0.5, dtype=torch.float32)
    # Moving low-contrast vertical fluid filament (advancing 1px per frame)
    fluid = base.clone()
    for t in range(5):
        col = 12 + t
        fluid[t, 6:26, col : col + 2, :] += 0.022

    # Compare uniform de-graining (detail_preservation=0.0) vs structure-preserving (0.80)
    cleaned_uniform = module._degrain_context_suffix(
        fluid, strength=0.65, detail_preservation=0.0
    )
    cleaned_preserved = module._degrain_context_suffix(
        fluid, strength=0.65, detail_preservation=0.80
    )
    filament_mask = fluid > 0.51
    loss_uniform = (fluid[filament_mask] - cleaned_uniform[filament_mask]).abs().mean().item()
    loss_preserved = (fluid[filament_mask] - cleaned_preserved[filament_mask]).abs().mean().item()
    assert loss_preserved < loss_uniform * 0.35

    # Verify 1px checkerboard noise is still scrubbed with detail_preservation=0.80
    checker = (
        (torch.arange(32).view(-1, 1) + torch.arange(32).view(1, -1)) % 2 * 2 - 1
    ).float() * 0.012
    noisy = base + checker.view(1, 32, 32, 1)
    cleaned_noisy = module._degrain_context_suffix(
        noisy, strength=0.35, detail_preservation=0.80
    )
    assert (cleaned_noisy - base).pow(2).mean() < (noisy - base).pow(2).mean() * 0.35


