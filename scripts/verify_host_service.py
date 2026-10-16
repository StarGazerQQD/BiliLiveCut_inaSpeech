"""验收真实宿主 DTO → 本机 FFmpeg → 独立服务 → 旁路分类的完整调用。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def main() -> None:
    """只使用宿主公共契约，不初始化或写入宿主数据库。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.host_source.resolve(strict=True)))
    from app.plugins import PluginContext
    from app.plugins.contracts import PluginSettingValue
    from app.plugins.highlight import HighlightScoringRequest

    from blc_inaspeech.adapter import Plugin
    from blc_inaspeech.cli import read_token

    media = args.media.resolve(strict=True)
    values: dict[str, PluginSettingValue] = {
        "endpoint": args.endpoint,
        "token": read_token(args.token_file),
        "media_root": str(media.parent),
    }

    def get(key: str, default: PluginSettingValue) -> PluginSettingValue:
        return values.get(key, default)

    def set_value(key: str, value: PluginSettingValue) -> None:
        values[key] = value

    plugin = Plugin()
    plugin.on_enable(PluginContext("blc-inaspeech", media.parent, get, set_value))
    started = datetime(2026, 1, 1)
    request = HighlightScoringRequest(
        1,
        1,
        1,
        started,
        started + timedelta(seconds=75),
        started,
        75.0,
        str(media),
        "integration fixture",
        None,
        None,
        None,
        None,
        (),
        (),
        None,
        0.42,
        room_mode="champion",
    )
    try:
        outcome = plugin.score_highlight(request)
        if outcome.uses_champion or outcome.champion_probability is not None or outcome.shadow_probability is not None:
            raise AssertionError("插件错误地输出或覆盖高光概率")
        if request.rule_score != 0.42 or request.transcript_text != "integration fixture":
            raise AssertionError("插件修改了宿主原始输入")
        if outcome.effective_mode != "shadow" or not outcome.feature_values:
            raise AssertionError("真实插件未产生旁路分类")
        if abs(sum(outcome.feature_values.values()) - 1) > 1e-6:
            raise AssertionError("分类占比无效")
        args.output.write_text(json.dumps({"status": "passed", "result": outcome.to_dict()}, indent=2) + "\n")
        print("真实宿主 DTO、Windows/Linux FFmpeg、独立服务和原评分保留验收通过。")
    finally:
        plugin.on_disable()


if __name__ == "__main__":
    main()
