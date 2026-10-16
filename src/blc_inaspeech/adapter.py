"""只依赖 BiliLiveCut 公共 API 的旁路分类插件。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.plugins import PluginContext, PluginSetting
from app.plugins.highlight import HighlightFeedback, HighlightScoringRequest, HighlightScoringResult

from blc_inaspeech.audio import decode_media, resolve_media
from blc_inaspeech.client import Client
from blc_inaspeech.schema import number

if TYPE_CHECKING:
    from collections.abc import Sequence

_FEATURE_NAMES = tuple(f"inaspeech_{label}_ratio" for label in ("speech", "music", "noise", "silence"))
_FINGERPRINT = hashlib.sha256(json.dumps(_FEATURE_NAMES, separators=(",", ":")).encode()).hexdigest()


class Plugin:
    """通过现有 highlight_scorer API 记录分类，不输出高光概率。"""

    settings_schema: Sequence[PluginSetting] = (
        PluginSetting(key="endpoint", label="推理服务地址", default="http://127.0.0.1:8766", required=True),
        PluginSetting(key="token", label="推理服务令牌", kind="password", default=""),
        PluginSetting(
            key="media_root",
            label="授权媒体根目录",
            default="",
            description="填写宿主可读取的录制/导入媒体目录；只允许分析该目录内的普通媒体文件。",
        ),
        PluginSetting(key="ffmpeg", label="FFmpeg 程序", default="ffmpeg", required=True),
        PluginSetting(key="timeout_s", label="推理等待秒数", kind="number", default=120, minimum=1, maximum=300),
        PluginSetting(key="decode_timeout_s", label="解码等待秒数", kind="number", default=30, minimum=1, maximum=120),
        PluginSetting(
            key="allow_network_http",
            label="允许专用网络明文 HTTP",
            kind="boolean",
            default=False,
            description="仅供受信任的 Docker 专用网络；公网服务请使用 HTTPS。",
        ),
    )

    def __init__(self) -> None:
        self.context: PluginContext | None = None

    def on_enable(self, context: PluginContext) -> None:
        """保存设置上下文；允许启用后进入设置，不隐式安装或下载模型。"""
        self.context = context

    def on_disable(self) -> None:
        """释放上下文；插件不创建进程、线程或后台任务。"""
        self.context = None

    def _text(self, key: str, default: str) -> str:
        if self.context is None:
            raise RuntimeError("inaSpeech 插件尚未启用")
        value = self.context.get_setting(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"请先配置 inaSpeech 插件设置：{key}")
        return value.strip()

    def score_highlight(self, request: HighlightScoringRequest) -> HighlightScoringResult:
        """完成解码与分类，保留宿主规则分和所有 ASR 行为。"""
        context = self.context
        if context is None:
            raise RuntimeError("inaSpeech 插件尚未启用")
        if request.room_mode == "off":
            return HighlightScoringResult(requested_mode="off", effective_mode="off")
        allow_http = context.get_setting("allow_network_http", False)
        if not isinstance(allow_http, bool):
            raise ValueError("allow_network_http 必须为布尔值")
        client = Client(
            self._text("endpoint", "http://127.0.0.1:8766"),
            self._text("token", ""),
            timeout_s=number(context.get_setting("timeout_s", 120), "timeout_s"),
            allow_network_http=allow_http,
        )
        media = resolve_media(Path(request.file_path), Path(self._text("media_root", "")))
        wav = decode_media(
            media,
            ffmpeg=self._text("ffmpeg", "ffmpeg"),
            timeout_s=number(context.get_setting("decode_timeout_s", 30), "decode_timeout_s"),
        )
        analysis = client.analyze(wav)
        return HighlightScoringResult(
            requested_mode="shadow",
            effective_mode="shadow",
            schema_version="inaspeech-1",
            schema_fingerprint=_FINGERPRINT,
            feature_values=analysis.features(),
            metadata={
                "audio_classification": analysis.to_dict(),
                "time_origin": "segment_media_start",
                "feedback_policy": "not_collected",
                "score_policy": "preserve_host_score",
                "room_requested_mode": request.room_mode,
            },
        )

    def record_highlight_feedback(self, feedback: HighlightFeedback) -> None:
        """分类模型不以高光审核结果训练，按公开契约明确不收集反馈。"""
        # 无副作用：不读取宿主数据库，也不创建包含用户转写或标签的训练文件。
