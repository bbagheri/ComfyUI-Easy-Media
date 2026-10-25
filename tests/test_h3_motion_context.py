from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.motion_context import core
from modules.motion_context import drift_control_av


class _NestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


@pytest.fixture
def nested_tensor_module(monkeypatch):
    comfy = sys.modules.get("comfy", types.ModuleType("comfy"))
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = _NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested_tensor)
    monkeypatch.setattr(comfy, "nested_tensor", nested_tensor, raising=False)


def _av_latent(
    *,
    video_steps: int = 7,
    height: int = 2,
    width: int = 2,
) -> dict:
    video = torch.arange(
        24 * video_steps * height * width,
        dtype=torch.float32,
    ).reshape(1, 24, video_steps, height, width)
    audio_steps = round(core._pixel_frames(video_steps) * core.FRAME_RESCALE)
    audio = torch.arange(
        32 * 2 * audio_steps,
        dtype=torch.float32,
    ).reshape(1, 32, 2, audio_steps)
    return {"samples": [video, audio]}


def _video_latent(
    *,
    video_steps: int = 2,
    height: int = 2,
    width: int = 2,
) -> dict:
    video = torch.arange(
        24 * video_steps * height * width,
        dtype=torch.float32,
    ).reshape(1, 24, video_steps, height, width)
    return {"samples": video}


def test_h3_motion_context_latent_path_uses_native_keyframes_and_keeps_refs():
    target = _av_latent()
    context = _av_latent()
    references = [{"kind": "image", "marker": "unchanged"}]

    output, trim_frames = core.apply_motion_context(
        conditioning=[[torch.tensor([1.0]), {"minimax_refs": references}]],
        vae=object(),
        latent=target,
        context_length="5",
        audio_context_length=3,
        context_latent=context,
    )

    metadata = output[0][1]
    assert trim_frames == 5
    assert len(metadata["minimax_keyframes"]) == 3
    video_keyframes = [
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("latent") is not None
    ]
    audio_keyframes = [
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("audio_latent") is not None
    ]
    assert [
        keyframe["resolved_frame_index"] for keyframe in video_keyframes
    ] == [0, 1]
    assert len(audio_keyframes) == 1
    assert audio_keyframes[0]["resolved_frame_index"] == pytest.approx(2.4)
    assert audio_keyframes[0]["audio_latent"].shape[-1] == 5
    assert metadata["minimax_refs"] is references
    assert "minimax_frame_count" not in metadata


def test_h3_motion_context_rejects_latent_resolution_change():
    with pytest.raises(ValueError, match="does not match target"):
        core.apply_motion_context(
            conditioning=[[torch.tensor([1.0]), {}]],
            vae=object(),
            latent=_av_latent(width=2),
            context_length="5",
            context_latent=_av_latent(width=3),
        )


def test_h3_hard_context_copies_matching_video_and_audio_windows(
    nested_tensor_module,
):
    target = _av_latent()
    context = _av_latent()

    output = core._hard_av_latent(
        target,
        context,
        video_frames=5,
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    output_video, output_audio = output["samples"].tensors
    context_video, context_audio = context["samples"]
    video_mask, audio_mask = output["noise_mask"].tensors
    assert isinstance(output["samples"], _NestedTensor)
    assert isinstance(output["noise_mask"], _NestedTensor)
    assert torch.equal(output_video[:, :, :2], context_video[:, :, -2:])
    assert torch.equal(output_audio[..., :8], context_audio[..., -8:])
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.allclose(
        video_mask[:, :, 1],
        torch.full_like(video_mask[:, :, 1], 0.5),
    )
    assert torch.all(audio_mask[..., :6] == 0)
    assert torch.all(audio_mask[..., 8:] == 1)


def test_h3_hard_context_merges_existing_masks_into_official_nested_tensor(
    nested_tensor_module,
):
    target = _av_latent()
    target_video, target_audio = target["samples"]
    target["noise_mask"] = [
        torch.full_like(target_video, 0.25),
        torch.full_like(target_audio, 0.4),
    ]

    output = core._hard_av_latent(
        target,
        _av_latent(),
        video_frames=5,
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    video_mask, audio_mask = output["noise_mask"].tensors
    assert isinstance(output["noise_mask"], _NestedTensor)
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.all(video_mask[:, :, 1:] == 0.25)
    assert torch.all(audio_mask[..., :6] == 0)
    assert torch.allclose(
        audio_mask[..., 6],
        torch.full_like(audio_mask[..., 6], 1 / 3),
    )
    assert torch.all(audio_mask[..., 7:] == 0.4)


def test_h3_reencoded_anchor_only_locks_last_five_frames_of_soft_context(
    nested_tensor_module,
):
    target = _av_latent(video_steps=12)
    target_video, target_audio = target["samples"]
    anchor = _video_latent(video_steps=2)
    anchor["samples"] = anchor["samples"] + 10_000

    output = core._reencoded_anchor_av_latent(
        target,
        anchor,
        context_frames=22,
        anchor_frames=5,
    )

    output_video, output_audio = output["samples"].tensors
    video_mask, audio_mask = output["noise_mask"].tensors
    assert torch.equal(output_video[:, :, :5], target_video[:, :, :5])
    assert torch.equal(output_video[:, :, 5:7], anchor["samples"])
    assert torch.equal(output_video[:, :, 7:], target_video[:, :, 7:])
    assert torch.all(video_mask[:, :, :5] == 1)
    assert torch.all(video_mask[:, :, 5:7] == 0)
    assert torch.all(video_mask[:, :, 7:] == 1)
    assert torch.equal(output_audio, target_audio)
    assert torch.all(audio_mask == 1)


def test_h3_reencoded_anchor_keeps_long_native_motion_context(
    nested_tensor_module,
):
    context = _av_latent(video_steps=7)
    context["anchor_samples"] = _video_latent(video_steps=2)["samples"]
    conditioning, trim_frames, latent = core.apply_reencoded_anchor_motion_context(
        conditioning=[[torch.tensor([1.0]), {}]],
        vae=object(),
        latent=_av_latent(video_steps=12),
        context_latent=context,
    )

    assert trim_frames == 22
    assert len(conditioning[0][1]["minimax_keyframes"]) == 8
    video_mask, _ = latent["noise_mask"].tensors
    assert torch.all(video_mask[:, :, :5] == 1)
    assert torch.all(video_mask[:, :, 5:7] == 0)


def test_h3_hires_reencoded_anchor_freezes_audio_and_only_video_boundary(
    nested_tensor_module,
):
    context = _av_latent(video_steps=7)
    context["anchor_samples"] = _video_latent(video_steps=2)["samples"]
    output, trim_frames = core.apply_hires_anchor_continuity(
        current_hires_latent=_av_latent(video_steps=12),
        previous_hires_latent=context,
    )

    video_mask, audio_mask = output["noise_mask"].tensors
    assert trim_frames == 22
    assert torch.all(video_mask[:, :, :5] == 1)
    assert torch.all(video_mask[:, :, 5:7] == 0)
    assert torch.all(video_mask[:, :, 7:] == 1)
    assert torch.all(audio_mask == 0)


def test_h3_anchor_falls_back_to_last_two_context_video_tokens():
    context = _av_latent(video_steps=7)
    anchor = core._video_anchor_from_context_latent(context)

    assert torch.equal(anchor["samples"], context["samples"][0][:, :, -2:])


def test_context_swap_noise_is_disposable_deterministic_and_video_only(
    nested_tensor_module,
):
    context = _av_latent()
    original_video, original_audio = context["samples"]
    original_video = original_video.clone()
    original_audio = original_audio.clone()

    first = core.apply_context_swap_noise(context, context_length="5", seed=42)
    second = core.apply_context_swap_noise(context, context_length="5", seed=42)
    first_video, first_audio = first["samples"].tensors
    second_video, second_audio = second["samples"].tensors

    assert torch.equal(context["samples"][0], original_video)
    assert torch.equal(context["samples"][1], original_audio)
    assert torch.equal(first_video[:, :, :5], original_video[:, :, :5])
    assert not torch.equal(first_video[:, :, 5:], original_video[:, :, 5:])
    assert torch.equal(first_video, second_video)
    assert torch.equal(first_audio, original_audio)
    assert torch.equal(second_audio, original_audio)


def test_context_swap_dynamic_mask_tracks_selflift_stage_resolution():
    state = drift_control_av._DriftControlMaskState(
        (1, 24, 7, 8, 10),
        torch.tensor([1.0, 0.5, 0.0]),
        prefix_steps=2,
    )
    active_shapes = [(1, 24, 7, 4, 6), (1, 32, 2, 20)]
    active_model = types.SimpleNamespace(
        inner_model=types.SimpleNamespace(latent_shapes=active_shapes)
    )
    packed_size = 24 * 7 * 4 * 6 + 32 * 2 * 20
    packed_mask = torch.ones(1, 1, packed_size)

    output = state.denoise_mask_function(
        torch.tensor(1.0),
        packed_mask,
        {"model": active_model},
    )

    assert output.shape == packed_mask.shape
    assert state.current_video_mask is not None
    assert state.current_video_mask.shape == (1, 1, 7, 4, 6)


def test_drift_control_reports_original_and_current_pixel_resolution():
    context = _av_latent(video_steps=7, height=2, width=3)
    target = _av_latent(video_steps=12, height=4, width=5)

    with pytest.raises(ValueError) as error:
        drift_control_av.prepare_context_swap_latent(
            target,
            context,
            5,
            continue_audio=True,
            freeze_audio=False,
        )

    assert "original context 48x32 pixels" in str(error.value)
    assert "current target 80x64 pixels" in str(error.value)
    assert "(1, 24, 2, 2, 3)" in str(error.value)
    assert "(1, 24, 12, 4, 5)" in str(error.value)


def test_drift_control_rejects_video_channel_mismatch_with_shapes():
    context = _av_latent(video_steps=7)
    context["samples"][0] = context["samples"][0][:, :23]

    with pytest.raises(ValueError, match=r"shapes differ: .*23.*24"):
        drift_control_av.prepare_context_swap_latent(
            _av_latent(video_steps=12),
            context,
            5,
            continue_audio=True,
            freeze_audio=False,
        )


def test_h3_hard_context_requires_native_video_and_audio_keyframes(monkeypatch):
    monkeypatch.setattr(
        core,
        "apply_motion_context",
        lambda **_kwargs: ([[torch.tensor([1.0]), {"minimax_keyframes": []}]], 5),
    )

    with pytest.raises(RuntimeError, match=r"Motion Context 0\.4\.0\+"):
        core.apply_hard_motion_context(
            conditioning=[[torch.tensor([1.0]), {}]],
            vae=object(),
            latent=_av_latent(),
            context_latent=_av_latent(),
            context_length="5",
        )


def test_h3_hard_context_keeps_native_audio_and_uses_it_for_hard_lock(
    nested_tensor_module,
):
    references = [{"kind": "audio", "marker": "user-reference"}]
    context = _av_latent()

    conditioning, trim_frames, latent = core.apply_hard_motion_context(
        conditioning=[[torch.tensor([1.0]), {"minimax_refs": references}]],
        vae=object(),
        latent=_av_latent(),
        context_latent=context,
        context_length="5",
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    metadata = conditioning[0][1]
    audio_keyframe = next(
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("audio_latent") is not None
    )
    _, hard_audio = latent["samples"].tensors
    assert trim_frames == 5
    assert metadata["minimax_refs"] is references
    assert torch.equal(
        hard_audio[..., : audio_keyframe["audio_latent"].shape[-1]],
        audio_keyframe["audio_latent"],
    )


def test_h3_hires_continuity_uses_previous_video_and_freezes_current_audio(
    nested_tensor_module,
):
    current = _av_latent()
    previous = _av_latent()

    output, trim_frames = core.apply_hires_continuity(
        current,
        previous,
        context_length="5",
        video_transition_steps=1,
    )

    output_video, output_audio = output["samples"].tensors
    previous_video, _ = previous["samples"]
    _, current_audio = current["samples"]
    video_mask, audio_mask = output["noise_mask"].tensors
    assert trim_frames == 5
    assert torch.equal(output_video[:, :, :2], previous_video[:, :, -2:])
    assert torch.equal(output_audio, current_audio)
    assert torch.all(video_mask[:, :, 2:] == 1)
    assert torch.all(audio_mask == 0)


def test_trim_motion_context_latent_keeps_only_detached_cpu_av_tail(
    nested_tensor_module,
):
    latent = _av_latent(video_steps=12)
    source_video, source_audio = latent["samples"]
    anchor_samples = _video_latent()["samples"]
    latent["anchor_samples"] = anchor_samples

    output = core.trim_motion_context_latent(latent, context_length="22")

    video, audio = output["samples"].tensors
    assert video.shape[2] == 7
    assert audio.shape[-1] == 37
    assert torch.equal(video, source_video[:, :, -7:])
    assert torch.equal(audio, source_audio[..., -37:])
    assert video.device.type == "cpu"
    assert audio.device.type == "cpu"
    assert torch.equal(output["anchor_samples"], anchor_samples)
    assert output["anchor_samples"].untyped_storage().data_ptr() != (
        anchor_samples.untyped_storage().data_ptr()
    )
    assert video.untyped_storage().data_ptr() != source_video.untyped_storage().data_ptr()
    assert audio.untyped_storage().data_ptr() != source_audio.untyped_storage().data_ptr()


def test_latent_rematerialization_spectral_reset_and_sacrificial_prefix_disruption():
    target = _av_latent(video_steps=12, height=8, width=8)
    context = _av_latent(video_steps=7, height=8, width=8)
    ref_latent = torch.full((1, 24, 1, 8, 8), -2.5, dtype=torch.float32)
    conditioning = [[torch.tensor([1.0]), {"minimax_refs": [{"kind": "image", "latent": ref_latent}]}]]

    # 1. With reset_denoise=0.0, all 7 video keyframes remain pinned
    legacy_cond, _ = core.apply_motion_context(
        conditioning=conditioning,
        vae=object(),
        latent=target,
        context_length="22",
        audio_context_length=0,
        context_latent=context,
        reset_denoise=0.0,
    )
    legacy_video_kf = [
        kf for kf in legacy_cond[0][1]["minimax_keyframes"] if kf.get("latent") is not None
    ]
    assert [kf["resolved_frame_index"] for kf in legacy_video_kf] == [0, 1, 5, 9, 13, 17, 18]

    # 2. With reset_denoise=0.30, intermediate sacrificial prefix tokens (t=1..4) are unpinned
    reset_cond, _ = core.apply_motion_context(
        conditioning=conditioning,
        vae=object(),
        latent=target,
        context_length="22",
        audio_context_length=0,
        context_latent=context,
        reset_denoise=0.30,
    )
    reset_video_kf = [
        kf for kf in reset_cond[0][1]["minimax_keyframes"] if kf.get("latent") is not None
    ]
    assert [kf["resolved_frame_index"] for kf in reset_video_kf] == [0, 17, 18]
    # Head sacrificial block (t=0) is pulled toward <Picture 1> ref_latent (-2.5)
    assert float(reset_video_kf[0]["latent"].mean()) < float(legacy_video_kf[0]["latent"].mean())


def test_drift_control_two_stage_sigma_release_eliminates_final_step_overwrite():
    sigmas = torch.tensor([1.0, 0.75, 0.50, 0.25, 0.10, 0.0])
    video_shape = (1, 24, 12, 8, 8)
    packed_size = 24 * 12 * 8 * 8 + 32 * 2 * 65
    packed_mask = torch.ones(1, 1, packed_size)

    state = drift_control_av._DriftControlMaskState(
        video_shape=video_shape,
        sigmas=sigmas,
        prefix_steps=7,
        seam_floor=0.30,
    )

    # Stage 1 (sigma=1.0 -> 0.75): terminal seam token (t=6, frames 18..21 before delivered frame 22)
    # is locked tightly (~0.14-0.16) while bridge token (t=5) sits near ~0.38
    state.denoise_mask_function(torch.tensor(1.0), packed_mask)
    bridge_high_sigma = float(state.current_video_mask[0, 0, 5, 0, 0])
    seam_high_sigma = float(state.current_video_mask[0, 0, 6, 0, 0])
    assert 0.30 <= bridge_high_sigma <= 0.48
    assert 0.12 <= seam_high_sigma <= 0.22

    # Stage 2 (sigma=0.25 -> 0.10): bridge token (t=5) ramps upward while terminal seam token (t=6)
    # rises gently toward its boundary-hold ceiling (~0.27)
    state.denoise_mask_function(torch.tensor(0.25), packed_mask)
    bridge_low_sigma = float(state.current_video_mask[0, 0, 5, 0, 0])
    seam_low_sigma = float(state.current_video_mask[0, 0, 6, 0, 0])
    assert bridge_low_sigma > bridge_high_sigma + 0.12
    assert seam_high_sigma < seam_low_sigma <= 0.32

    # Final step (sigma=0.10 -> 0.0): sacrificial prefix tokens (t=0..4) reach full release (1.0),
    # bridge token (t=5) reaches ~0.59, and terminal seam token (t=6) holds at ~0.27 to prevent any jump!
    state.denoise_mask_function(torch.tensor(0.10), packed_mask)
    assert torch.allclose(
        state.current_video_mask[:, :, :5],
        torch.ones_like(state.current_video_mask[:, :, :5]),
    )
    assert 0.55 <= float(state.current_video_mask[0, 0, 5, 0, 0]) <= 0.75
    assert 0.24 <= float(state.current_video_mask[0, 0, 6, 0, 0]) <= 0.32


def test_hires_anchor_continuity_second_pass_blends_pass1_and_boosts_release_mask(
    nested_tensor_module,
):
    current = _av_latent(video_steps=12, height=8, width=8)
    previous = _av_latent(video_steps=7, height=8, width=8)
    previous["anchor_samples"] = torch.full((1, 24, 2, 8, 8), 50.0, dtype=torch.float32)

    output, trim_frames = core.apply_hires_anchor_continuity(
        current_hires_latent=current,
        previous_hires_latent=previous,
        context_length="22",
        anchor_length="5",
        reset_denoise=0.30,
    )
    video_mask, audio_mask = output["noise_mask"].tensors
    assert trim_frames == 22
    # In pass 2 at reset_denoise=0.30, bridge token (t=5, frame 17) releases above 0.60,
    # while terminal seam token (t=6, frames 18..21 before delivered frame 22) stays tightly
    # locked (~0.27) so delivered frame 22 splices seamlessly with zero boundary jump.
    assert float(video_mask[0, 0, 5, 0, 0]) > 0.60
    assert 0.24 <= float(video_mask[0, 0, 6, 0, 0]) <= 0.35
    assert float(video_mask[0, 0, 6, 0, 0]) < float(video_mask[0, 0, 5, 0, 0])
    assert torch.all(audio_mask == 0)
