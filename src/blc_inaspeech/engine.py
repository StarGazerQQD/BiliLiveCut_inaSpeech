"""固定官方模型、常驻实例、有限内存缓存的串行分类引擎。"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from blc_inaspeech.audio import wav_duration
from blc_inaspeech.models import model_manifest, verify_model
from blc_inaspeech.schema import Analysis, Label, Segment, number


class Predictor(Protocol):
    """外部模型边界，便于以真实 HTTP/文件测试其余处理逻辑。"""

    def __call__(self, path: str) -> Sequence[tuple[str, float, float]]:
        """对规范的本地 WAV 返回上游分类区间。"""
        ...


def load_predictor(model_dir: Path, *, batch_size: int = 128, device: str = "cpu", threads: int = 2) -> Predictor:
    """在独立进程中初始化 TensorFlow，禁用运行时模型下载。"""
    if device not in {"cpu", "cuda"} or not 1 <= batch_size <= 1024 or not 1 <= threads <= 32:
        raise ValueError("无效的推理设备、批量大小或线程数")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    verified = verify_model(model_dir, "keras_speech_music_noise_cnn.hdf5")
    import tensorflow as tf
    from inaSpeechSegmenter import segmenter as upstream

    if device == "cuda":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("请求 CUDA，但 TensorFlow 未检测到 GPU；拒绝静默降为 CPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def verified_remote(name: str) -> str:
        # 只替换本进程的模型定位边界，不修改上游源码或 CNN/Viterbi。
        if name != verified.name:
            raise ValueError("生产服务不加载性别或未登记模型")
        return str(verify_model(model_dir, name))

    upstream.get_remote = verified_remote
    return upstream.Segmenter(vad_engine="smn", detect_gender=False, ffmpeg=None, batch_size=batch_size)  # type: ignore[no-any-return]


def normalize_segments(raw: Sequence[tuple[str, float, float]], duration: float) -> tuple[Segment, ...]:
    """把上游 20ms 栅格裁到真实长度，最多补齐末尾一帧误差。"""
    labels: dict[str, Label] = {"speech": "speech", "music": "music", "noise": "noise", "noEnergy": "silence"}
    result: list[Segment] = []
    cursor = 0.0
    for label, start_raw, end_raw in raw:
        start = number(start_raw, "start")
        end = number(end_raw, "end")
        if label not in labels or abs(start - cursor) > 1e-7 or not start < end or end > duration + 0.04:
            raise ValueError("上游分类区间不连续或标签/边界无效")
        if start >= duration:
            break
        clipped = min(end, duration)
        result.append(Segment(labels[label], start, clipped))
        cursor = end
    if not result or abs(result[-1].end_s - duration) > 0.04:
        raise ValueError("上游分类未覆盖音频")
    result[-1] = replace(result[-1], end_s=duration)
    return tuple(result)


class Engine:
    """单调用者使用的常驻引擎；缓存仅保存结果，不保存音频。"""

    def __init__(self, predictor: Predictor, *, cache_entries: int = 32) -> None:
        if not 0 <= cache_entries <= 128:
            raise ValueError("缓存数量必须为 0 至 128")
        self.predictor = predictor
        self.cache_entries = cache_entries
        self.model_hash = model_manifest()["keras_speech_music_noise_cnn.hdf5"].sha256
        self._cache: OrderedDict[str, Analysis] = OrderedDict()

    def analyze(self, data: bytes) -> Analysis:
        """按完整音频内容命中缓存，避免路径和 mtime 缓存失效漏洞。"""
        duration = wav_duration(data)
        key = hashlib.sha256(data).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return replace(cached, cache_hit=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="blc-ina-worker-") as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(data)
            raw = self.predictor(str(path))
        analysis = Analysis(duration, normalize_segments(raw, duration), self.model_hash, time.perf_counter() - started)
        # 对外部模型输出使用与客户端相同的协议验收。
        Analysis.from_dict(analysis.to_dict(), expected_model=self.model_hash)
        if self.cache_entries:
            self._cache[key] = analysis
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return analysis
