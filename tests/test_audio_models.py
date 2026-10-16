from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blc_inaspeech.audio import decode_media, resolve_media, wav_duration
from blc_inaspeech.cli import create_token, read_token
from blc_inaspeech.models import verify_model


def test_real_ffmpeg_and_path_boundary(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "有 空格;$(not-a-command).wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x01\x00" * 8000)
    assert resolve_media(path, tmp_path) == path.resolve()
    assert wav_duration(decode_media(path)) == 1.0
    with pytest.raises(ValueError):
        resolve_media(path, tmp_path / "other")
    forbidden = tmp_path / "secret.txt"
    forbidden.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_media(forbidden, tmp_path)
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        resolve_media(outside, tmp_path)


def test_bad_media_and_decode_timeout_validation(tmp_path: Path) -> None:
    media = tmp_path / "bad.mp4"
    media.write_bytes(b"not-media")
    with pytest.raises(RuntimeError, match="FFmpeg"):
        decode_media(media)
    with pytest.raises(ValueError):
        decode_media(media, timeout_s=0)


def test_token_unique_no_overwrite_and_invalid_file(tmp_path: Path) -> None:
    a, b = tmp_path / "a.token", tmp_path / "b.token"
    create_token(a)
    create_token(b)
    assert len(read_token(a)) == 64
    assert read_token(a) != read_token(b)
    original = a.read_bytes()
    with pytest.raises(FileExistsError):
        create_token(a)
    assert a.read_bytes() == original
    b.write_text("not-a-secret", encoding="ascii")
    with pytest.raises(ValueError):
        read_token(b)


def test_model_tamper_and_unknown_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        verify_model(tmp_path, "../../secret")
    name = "keras_speech_music_noise_cnn.hdf5"
    path = tmp_path / name
    path.write_bytes(b"x" * 3244808)
    expected = "f04b5e3c86fa2e81666d106b0867350fe7858c7bd006c2a74c27b5b45507e7d8"
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected
    with pytest.raises(ValueError, match="SHA-256"):
        verify_model(tmp_path, name)
