import sys
import os
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folder_paths
from utils.media import allowed_extensions, delete_media_file, file_entry, list_recent_media_files
from utils.multitrack import (
    image_data_files_signature,
    multitrack_media_files_signature,
    multitrack_segment_media_signature,
)


def test_file_entry_encodes_unicode_view_url_components(tmp_path):
    subfolder = tmp_path / "中文 子目录"
    subfolder.mkdir()
    filename = (
        "jimeng-2026-02-13-9387-基于经典“子弹时间”段落进行镜头语言级复刻，"
        "镜头角度、景别、节奏与原片保持一致，....mp4"
    )
    media_file = subfolder / filename
    media_file.write_bytes(b"video")

    entry = file_entry(
        str(media_file),
        f"中文 子目录/{filename}",
        "inputs",
    )

    assert entry["url"] == (
        "/view?filename=jimeng-2026-02-13-9387-%E5%9F%BA%E4%BA%8E%E7%BB%8F%E5%85%B8"
        "%E2%80%9C%E5%AD%90%E5%BC%B9%E6%97%B6%E9%97%B4%E2%80%9D%E6%AE%B5%E8%90%BD"
        "%E8%BF%9B%E8%A1%8C%E9%95%9C%E5%A4%B4%E8%AF%AD%E8%A8%80%E7%BA%A7%E5%A4%8D"
        "%E5%88%BB%EF%BC%8C%E9%95%9C%E5%A4%B4%E8%A7%92%E5%BA%A6%E3%80%81%E6%99%AF"
        "%E5%88%AB%E3%80%81%E8%8A%82%E5%A5%8F%E4%B8%8E%E5%8E%9F%E7%89%87%E4%BF%9D"
        "%E6%8C%81%E4%B8%80%E8%87%B4%EF%BC%8C....mp4"
        "&type=input&subfolder=%E4%B8%AD%E6%96%87%20%E5%AD%90%E7%9B%AE%E5%BD%95"
    )


def test_file_entry_supports_temp_preview_urls(tmp_path):
    media_file = tmp_path / "easy_compare_source_preview.mp4"
    media_file.write_bytes(b"video")

    entry = file_entry(str(media_file), "easy_compare_source_preview.mp4", "temp")

    assert entry["url"] == "/view?filename=easy_compare_source_preview.mp4&type=temp&subfolder="


def test_list_recent_media_files_recurses_and_filters_by_recency_and_type(tmp_path):
    output_dir = tmp_path / "output"
    nested = output_dir / "sub" / "nested"
    nested.mkdir(parents=True)

    recent = nested / "recent.mp4"
    old = output_dir / "old.mp4"
    recent.write_bytes(b"recent")
    old.write_bytes(b"old")

    now = time.time()
    os.utime(recent, (now, now))
    os.utime(old, (now - 3 * 24 * 3600, now - 3 * 24 * 3600))

    items = list_recent_media_files(
        output_dir,
        "",
        "outputs",
        allowed_extensions("video"),
        hours=48,
        limit=100,
    )

    assert [item["path"] for item in items] == [str(Path("sub") / "nested" / "recent.mp4")]


def test_delete_media_file_removes_allowed_file_and_blocks_traversal(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    img = input_dir / "office4.png"
    img.write_bytes(b"png-v1")

    deleted = delete_media_file(str(input_dir), "office4.png", allowed_extensions("all"))
    assert deleted is True
    assert not img.exists()

    with pytest.raises(ValueError, match="outside base directory"):
        delete_media_file(str(input_dir), "../outside.png", allowed_extensions("all"))


def test_multitrack_media_signatures_change_when_input_file_overwritten_in_place(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_dir))

    img = input_dir / "office4.png"
    img.write_bytes(b"png-version-1")
    os.utime(img, (1700000000, 1700000000))

    tracks_info = {
        "tracks": [
            {
                "id": "track-task",
                "type": "task",
                "segments": [
                    {
                        "id": "seg-1",
                        "start_frame": 0,
                        "end_frame": 80,
                        "content": {
                            "type": "task",
                            "images": [
                                {
                                    "id": "img-1",
                                    "file_name": "office4.png",
                                    "source_type": "input",
                                    "file_path": "office4.png",
                                    "shared_reference": True,
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    sig1 = multitrack_media_files_signature(tracks_info)
    seg_sig1 = multitrack_segment_media_signature(tracks_info, 0)
    assert sig1 != ""
    assert seg_sig1 != ""

    # Overwrite the same file in-place with updated bytes and mtime
    img.write_bytes(b"png-version-2-updated-content")
    os.utime(img, (1700000010, 1700000010))

    sig2 = multitrack_media_files_signature(tracks_info)
    seg_sig2 = multitrack_segment_media_signature(tracks_info, 0)

    assert sig1 != sig2
    assert seg_sig1 != seg_sig2


def test_image_data_files_signature_changes_when_input_file_overwritten(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_dir))

    img = input_dir / "hero.png"
    img.write_bytes(b"v1")
    os.utime(img, (1700000000, 1700000000))

    payload = '{"images": [{"source_type": "input", "file_path": "hero.png"}]}'
    sig1 = image_data_files_signature(payload)
    assert sig1 != ""

    img.write_bytes(b"v2-modified")
    os.utime(img, (1700000005, 1700000005))

    sig2 = image_data_files_signature(payload)
    assert sig1 != sig2
