"""使用固定音频和真实官方模型串行比较常驻、batch、性别开关及缓存。"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

from blc_inaspeech.audio import wav_duration
from blc_inaspeech.engine import Engine, Predictor, load_predictor, normalize_segments
from blc_inaspeech.models import verify_model
from blc_inaspeech.schema import Segment

R = TypeVar("R")


def canonical(raw: Sequence[tuple[str, float, float]], duration: float) -> tuple[Segment, ...]:
    """只在基线比较中将性别标签合并为 speech，不改变时间边界。"""
    merged: list[tuple[str, float, float]] = []
    for label, start, end in raw:
        label = "speech" if label in {"female", "male"} else label
        if merged and merged[-1][0] == label:
            previous = merged[-1]
            merged[-1] = (label, previous[1], end)
        else:
            merged.append((label, start, end))
    return normalize_segments(merged, duration)


def measure(operation: Callable[[], R], iterations: int) -> tuple[list[float], list[R]]:
    """测量多次真实操作并保留所有结果，禁止合并不同口径的指标。"""
    samples: list[float] = []
    results: list[R] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
        results.append(result)
    return samples, results


def main() -> None:
    """记录样本身份、环境、原始耗时及精确分类一致性。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.iterations <= 20:
        raise ValueError("性能测量至少 3 次，最多 20 次")
    data = args.wav.read_bytes()
    duration = wav_duration(data)
    load_started = time.perf_counter()
    resident128 = load_predictor(args.model_dir, batch_size=128, device="cpu", threads=2)
    first_load = time.perf_counter() - load_started
    reference = canonical(resident128(str(args.wav)), duration)
    from inaSpeechSegmenter import segmenter as upstream

    def factory(batch: int, gender: bool = False) -> Predictor:
        def locate(name: str) -> str:
            return str(verify_model(args.model_dir, name))

        upstream.get_remote = locate
        return upstream.Segmenter(vad_engine="smn", detect_gender=gender, ffmpeg=None, batch_size=batch)  # type: ignore[no-any-return]

    scenarios: dict[str, dict[str, object]] = {}

    def record(name: str, operation: Callable[[], Sequence[tuple[str, float, float]]]) -> None:
        samples, results = measure(operation, args.iterations)
        if any(canonical(result, duration) != reference for result in results):
            raise AssertionError(f"分类区间发生变化：{name}")
        scenarios[name] = {
            "seconds": samples,
            "median_s": statistics.median(samples),
            "rtf": statistics.median(samples) / duration,
            "same_segments": True,
        }

    def reload_each_time() -> Sequence[tuple[str, float, float]]:
        model = factory(32)
        return model(str(args.wav))

    record("reload_model_batch32_no_gender", reload_each_time)
    gc.collect()
    resident32 = factory(32)
    resident32(str(args.wav))
    record("resident_batch32_no_gender", partial(resident32, str(args.wav)))
    del resident32
    gc.collect()
    record("resident_batch128_no_gender", lambda: resident128(str(args.wav)))
    resident_gender = factory(128, True)
    resident_gender(str(args.wav))
    record("resident_batch128_with_gender", partial(resident_gender, str(args.wav)))
    del resident_gender
    gc.collect()

    cache = Engine(resident128, cache_entries=1)
    first = cache.analyze(data)
    hit_samples, cached = measure(lambda: cache.analyze(data), args.iterations)
    if first.cache_hit or not all(result.cache_hit and result.segments == first.segments for result in cached):
        raise AssertionError("缓存对比未实际命中")

    profile = cProfile.Profile()
    profile.enable()
    resident128(str(args.wav))
    profile.disable()
    hotspots = sorted(profile.getstats(), key=lambda row: row.totaltime, reverse=True)[:20]
    report = {
        "sample": {"name": args.wav.name, "sha256": hashlib.sha256(data).hexdigest(), "duration_s": duration},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": version("tensorflow"),
            "inaspeech": version("inaSpeechSegmenter"),
            "device": "cpu",
            "threads": 2,
            "iterations": args.iterations,
        },
        "first_import_and_model_load_s": first_load,
        "scenarios": scenarios,
        "cache_hit": {
            "seconds": hit_samples,
            "median_s": statistics.median(hit_samples),
            "scope": "same WAV bytes; not new-audio inference",
        },
        "profile": [
            {
                "file": "<native>" if isinstance(entry.code, str) else Path(entry.code.co_filename).name,
                "function": entry.code if isinstance(entry.code, str) else entry.code.co_name,
                "calls": entry.callcount,
                "self_s": entry.inlinetime,
                "cumulative_s": entry.totaltime,
            }
            for entry in hotspots
        ],
        "limitations": [
            "single public mixed-audio sample; no Chinese accuracy benchmark",
            "same-process reload baseline excludes repeated TensorFlow import",
            "Docker Windows bind mount; hardware-dependent; no CUDA measurements",
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("真实性能测量通过，所有基线的合并分类区间一致。")


if __name__ == "__main__":
    main()
