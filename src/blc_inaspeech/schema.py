"""对不受信任的推理结果执行有限值、边界、覆盖与大小校验。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

Label = Literal["speech", "music", "noise", "silence"]
LABELS: tuple[Label, ...] = ("speech", "music", "noise", "silence")
MAX_DURATION = 600.0
MAX_WAV_BYTES = 19_201_024
MAX_RESPONSE_BYTES = 2_000_000
MAX_SEGMENTS = 30_001
SCHEMA_VERSION = "1"


def number(value: object, name: str) -> float:
    """读取有限实数，不接受布尔值或隐式字符串转换。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    return float(value)


@dataclass(frozen=True, slots=True)
class Segment:
    """以当前片段起点为零的半开秒区间。"""

    label: Label
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, object]:
        """输出公开 JSON 格式。"""
        return {"label": self.label, "start_s": self.start_s, "end_s": self.end_s}


@dataclass(frozen=True, slots=True)
class Analysis:
    """经过验证的完整分类结果，分类占比不代表高光概率。"""

    duration_s: float
    segments: tuple[Segment, ...]
    model_sha256: str
    inference_s: float
    cache_hit: bool = False

    def to_dict(self) -> dict[str, object]:
        """输出有明确版本和模型身份的结果。"""
        return {
            "schema_version": SCHEMA_VERSION,
            "engine": "inaSpeechSegmenter",
            "engine_version": "0.8.0",
            "duration_s": self.duration_s,
            "segments": [segment.to_dict() for segment in self.segments],
            "model_sha256": self.model_sha256,
            "inference_s": self.inference_s,
            "cache_hit": self.cache_hit,
        }

    def features(self) -> dict[str, float]:
        """在线性时间内汇总各分类的时长占比。"""
        durations = dict.fromkeys(LABELS, 0.0)
        for segment in self.segments:
            durations[segment.label] += segment.end_s - segment.start_s
        return {f"inaspeech_{label}_ratio": duration / self.duration_s for label, duration in durations.items()}

    @classmethod
    def from_dict(cls, payload: object, *, expected_model: str) -> Analysis:
        """拒绝缺失、额外字段、模型不符、重叠或未覆盖的区间。"""
        expected = {
            "schema_version",
            "engine",
            "engine_version",
            "duration_s",
            "segments",
            "model_sha256",
            "inference_s",
            "cache_hit",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("分类结果字段不符合协议")
        if (payload["schema_version"], payload["engine"], payload["engine_version"]) != (
            SCHEMA_VERSION,
            "inaSpeechSegmenter",
            "0.8.0",
        ):
            raise ValueError("分类引擎或协议版本不匹配")
        if payload["model_sha256"] != expected_model:
            raise ValueError("分类模型身份不匹配")
        duration = number(payload["duration_s"], "duration_s")
        elapsed = number(payload["inference_s"], "inference_s")
        if not 0.72 <= duration <= MAX_DURATION or elapsed < 0 or not isinstance(payload["cache_hit"], bool):
            raise ValueError("分类时长、耗时或缓存状态无效")
        raw = payload["segments"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SEGMENTS:
            raise ValueError("分类区间数量无效")
        segments: list[Segment] = []
        cursor = 0.0
        for item in raw:
            if not isinstance(item, dict) or set(item) != {"label", "start_s", "end_s"}:
                raise ValueError("分类区间字段无效")
            label = item["label"]
            if label not in LABELS:
                raise ValueError("未知音频分类")
            start = number(item["start_s"], "start_s")
            end = number(item["end_s"], "end_s")
            if abs(start - cursor) > 1e-7 or not start < end <= duration + 1e-7:
                raise ValueError("音频分类区间未连续覆盖或越界")
            segments.append(Segment(cast(Label, label), start, end))
            cursor = end
        if abs(cursor - duration) > 1e-7:
            raise ValueError("音频分类未覆盖整个片段")
        return cls(duration, tuple(segments), expected_model, elapsed, payload["cache_hit"])
