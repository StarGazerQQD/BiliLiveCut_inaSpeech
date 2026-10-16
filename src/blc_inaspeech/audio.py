"""有界本地媒体解码与标准 PCM WAV 验证。"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

from blc_inaspeech.schema import MAX_DURATION, MAX_WAV_BYTES

_MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".flv", ".ts", ".mov", ".mp3", ".ogg", ".flac", ".wav", ".m4a"}
_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,flv,mpegts,wav,mp3,ogg,flac,aac"


def wav_duration(data: bytes) -> float:
    """只接受完整、16 kHz、单声道、16-bit、长度有界的 PCM WAV。"""
    if not 44 <= len(data) <= MAX_WAV_BYTES:
        raise ValueError("WAV 大小越界")
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (
                1,
                2,
                16_000,
                "NONE",
            ):
                raise ValueError("仅接受 16 kHz 单声道 16-bit PCM WAV")
            frames = reader.getnframes()
            duration = frames / 16_000
            if not 0.72 <= duration <= MAX_DURATION:
                raise ValueError("音频长度必须为 0.72 至 600 秒")
            if len(reader.readframes(frames + 1)) != frames * 2:
                raise ValueError("WAV 音频数据不完整")
    except (wave.Error, EOFError) as exc:
        raise ValueError("WAV 格式无效") from exc
    return duration


def resolve_media(media: Path, root: Path) -> Path:
    """限制宿主只解码管理员指定目录中的普通媒体文件。"""
    try:
        root = root.expanduser().resolve(strict=True)
        resolved = media.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("授权目录或媒体文件不存在或不可访问") from exc
    if not root.is_dir() or media.is_symlink():
        raise ValueError("媒体根目录或文件无效")
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in _MEDIA_SUFFIXES:
        raise ValueError("媒体文件越出授权目录或格式不支持")
    # 也拒绝中间目录符号链接，避免意外访问被链接的素材目录。
    cursor = media.expanduser().absolute()
    while cursor != root and root in cursor.parents:
        if cursor.is_symlink():
            raise ValueError("媒体路径不得包含符号链接")
        cursor = cursor.parent
    return resolved


def decode_media(media: Path, *, ffmpeg: str = "ffmpeg", timeout_s: float = 30.0) -> bytes:
    """通过无 shell、限协议、限解码时长的 FFmpeg 得到规范 WAV。"""
    if not 1 <= timeout_s <= 120:
        raise ValueError("解码超时必须为 1 至 120 秒")
    with tempfile.TemporaryDirectory(prefix="blc-ina-") as directory:
        output = Path(directory) / "audio.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            _FORMATS,
            "-threads",
            "1",
            "-i",
            str(media.resolve(strict=True)),
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-bitexact",
            "-vn",
            "-sn",
            "-dn",
            "-t",
            str(MAX_DURATION + 0.1),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output),
        ]
        # 丢弃上游错误文本，避免在宿主日志中泄露路径或媒体内嵌信息。
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError("FFmpeg 音频解码失败；请检查媒体是否有有效音轨")
        if output.stat().st_size > MAX_WAV_BYTES:
            raise ValueError("媒体超过 600 秒限制")
        data = output.read_bytes()
    wav_duration(data)
    return data
