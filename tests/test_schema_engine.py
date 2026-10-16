from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blc_inaspeech.audio import wav_duration
from blc_inaspeech.engine import Engine, Predictor, load_predictor, normalize_segments
from blc_inaspeech.models import model_manifest
from blc_inaspeech.schema import Analysis, Segment


def payload() -> dict[str, object]:
    return Analysis(
        1.0,
        (Segment("speech", 0, 0.4), Segment("music", 0.4, 1.0)),
        model_manifest()["keras_speech_music_noise_cnn.hdf5"].sha256,
        0.1,
    ).to_dict()


def test_roundtrip_features_and_no_probability() -> None:
    data = payload()
    result = Analysis.from_dict(data, expected_model=str(data["model_sha256"]))
    assert result.to_dict() == data
    assert result.features() == {
        "inaspeech_speech_ratio": 0.4,
        "inaspeech_music_ratio": 0.6,
        "inaspeech_noise_ratio": 0.0,
        "inaspeech_silence_ratio": 0.0,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("duration_s", True),
        ("duration_s", float("nan")),
        ("duration_s", 601),
        ("duration_s", "1"),
        ("inference_s", -1),
        ("inference_s", float("inf")),
        ("cache_hit", "false"),
        ("model_sha256", "wrong"),
        ("engine_version", "0.7.8"),
        ("segments", []),
        ("segments", [{"label": "male", "start_s": 0, "end_s": 1}]),
        ("segments", [{"label": "speech", "start_s": 0.1, "end_s": 1}]),
        ("segments", [{"label": "speech", "start_s": 0, "end_s": 0.9}]),
        ("segments", [{"label": "speech", "start_s": 0, "end_s": 1.1}]),
        ("segments", [{"label": "speech", "start_s": 0, "end_s": True}]),
    ],
)
def test_invalid_results_rejected(key: str, value: object) -> None:
    data = copy.deepcopy(payload())
    expected = str(data["model_sha256"])
    data[key] = value
    with pytest.raises(ValueError):
        Analysis.from_dict(data, expected_model=expected)


def test_extra_fields_and_overlapping_intervals_rejected() -> None:
    data = payload()
    expected = str(data["model_sha256"])
    data["unexpected"] = 1
    with pytest.raises(ValueError):
        Analysis.from_dict(data, expected_model=expected)
    del data["unexpected"]
    data["segments"] = [
        {"label": "speech", "start_s": 0, "end_s": 0.6},
        {"label": "music", "start_s": 0.5, "end_s": 1.0},
    ]
    with pytest.raises(ValueError):
        Analysis.from_dict(data, expected_model=expected)


def test_normalize_upstream_grid_and_silence() -> None:
    result = normalize_segments([("noEnergy", 0, 0.5), ("speech", 0.5, 0.98)], 1.0)
    assert result == (Segment("silence", 0, 0.5), Segment("speech", 0.5, 1.0))
    assert normalize_segments([("music", 0, 1.02)], 1.0) == (Segment("music", 0, 1),)


@pytest.mark.parametrize("raw", [[], [("male", 0, 1)], [("speech", 0.1, 1)], [("speech", 0, 0.5)]])
def test_normalize_rejects_missing_or_invalid_model_results(raw: list[tuple[str, float, float]]) -> None:
    with pytest.raises(ValueError):
        normalize_segments(raw, 1.0)


@pytest.mark.parametrize("duration,rate,channels", [(0.71, 16000, 1), (1, 8000, 1), (1, 16000, 2)])
def test_invalid_wav_rejected(make_wav: Callable[..., bytes], duration: float, rate: int, channels: int) -> None:
    with pytest.raises(ValueError):
        wav_duration(make_wav(duration, rate=rate, channels=channels))


def test_truncated_wav_rejected(make_wav: Callable[..., bytes]) -> None:
    data = make_wav()
    with pytest.raises(ValueError):
        wav_duration(data[:-100])


def test_cache_content_key_lru_and_disabled(make_wav: Callable[..., bytes], predictor: Predictor) -> None:
    engine = Engine(predictor, cache_entries=1)
    a, b = make_wav(), make_wav(frequency=300)
    first = engine.analyze(a)
    cached = engine.analyze(a)
    assert not first.cache_hit and cached.cache_hit
    assert cached.segments == first.segments
    assert not engine.analyze(b).cache_hit
    assert not engine.analyze(a).cache_hit
    disabled = Engine(predictor, cache_entries=0)
    assert not disabled.analyze(a).cache_hit
    assert not disabled.analyze(a).cache_hit


def test_cuda_mode_refuses_missing_gpu_instead_of_starting_cpu_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tensorflow = ModuleType("tensorflow")
    monkeypatch.setattr(tensorflow, "config", SimpleNamespace(list_physical_devices=lambda kind: []), raising=False)
    upstream = ModuleType("inaSpeechSegmenter")

    def forbidden_model(**options: object) -> None:
        pytest.fail("CUDA 缺失时不应构造 CPU 或其他模型")

    monkeypatch.setattr(upstream, "segmenter", SimpleNamespace(Segmenter=forbidden_model), raising=False)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setitem(sys.modules, "inaSpeechSegmenter", upstream)
    monkeypatch.setattr("blc_inaspeech.engine.verify_model", lambda root, name: Path(name))
    with pytest.raises(RuntimeError, match="拒绝静默降为 CPU"):
        load_predictor(Path("models"), device="cuda")
