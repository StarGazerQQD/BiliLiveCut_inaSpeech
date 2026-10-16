from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from blc_inaspeech.engine import Engine, Predictor


def test_actual_manager_load_settings_scoring_and_shutdown(
    host_source: Path,
    tmp_path: Path,
    running_server: Callable[[Predictor], tuple[str, str, Engine]],
    predictor: Predictor,
    make_wav: Callable[..., bytes],
) -> None:
    from app.db.session import init_db
    from app.plugins.highlight import HighlightScoringRequest
    from app.plugins.manager import PluginManager

    init_db()
    endpoint, token, _ = running_server(predictor)
    plugins = tmp_path / "plugins"
    directory = plugins / "blc-inaspeech"
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "plugin/blc-inaspeech", directory)
    media = tmp_path / "sample.wav"
    media.write_bytes(make_wav())
    started = datetime(2026, 1, 1)
    request = HighlightScoringRequest(
        segment_id=1,
        session_id=1,
        room_id=1,
        start_ts=started,
        end_ts=started + timedelta(seconds=1),
        session_started_at=started,
        duration_s=1,
        file_path=str(media),
        transcript_text="已有转写",
        words=None,
        asr_avg_logprob=None,
        asr_review_risk=None,
        auxiliary=None,
        window_danmaku=(),
        baseline_danmaku=(),
        audio=None,
        rule_score=0.42,
        room_mode="champion",
    )

    async def scenario() -> None:
        manager = PluginManager(root=plugins)
        await manager.start()
        assert not manager.list_payload()["scan_errors"]
        await manager.set_enabled("blc-inaspeech", True)
        manager.update_settings("blc-inaspeech", {"endpoint": endpoint, "token": token, "media_root": str(tmp_path)})
        dispatch = manager.score_highlight(request)
        assert dispatch is not None and dispatch.error is None
        result = dispatch.prediction
        assert result is not None and result.effective_mode == "shadow"
        assert not result.uses_champion
        assert result.champion_probability is None and result.shadow_probability is None
        assert result.feature_values["inaspeech_speech_ratio"] == 0.5
        assert result.metadata["score_policy"] == "preserve_host_score"
        assert request.rule_score == 0.42 and request.transcript_text == "已有转写"
        assert json.loads(json.dumps(result.to_dict()))["metadata"]["audio_classification"]["segments"]
        # 密码不在公开设置回显中。
        assert token not in json.dumps(manager.settings_payload("blc-inaspeech"))
        manager.update_settings("blc-inaspeech", {"token": ""})
        assert manager.score_highlight(request).error is None
        await manager.set_enabled("blc-inaspeech", False)
        assert manager.score_highlight(request) is None
        await manager.stop()
        restarted = PluginManager(root=plugins)
        await restarted.start()
        assert not restarted.has_capability("highlight_scorer")
        await restarted.stop()

    asyncio.run(scenario())


def test_adapter_off_and_invalid_service_do_not_change_host_result(
    host_source: Path,
    tmp_path: Path,
) -> None:
    from app.plugins import PluginContext
    from app.plugins.highlight import HighlightScoringRequest

    from blc_inaspeech.adapter import Plugin

    plugin = Plugin()
    plugin.on_enable(PluginContext("blc-inaspeech", tmp_path, lambda key, default: default, lambda key, value: None))
    started = datetime(2026, 1, 1)
    request = HighlightScoringRequest(
        1,
        1,
        1,
        started,
        started + timedelta(seconds=1),
        started,
        1,
        "missing.mp4",
        None,
        None,
        None,
        None,
        None,
        (),
        (),
        None,
        0.77,
        room_mode="off",
    )
    result = plugin.score_highlight(request)
    assert result.requested_mode == "off" and not result.uses_champion
    request = replace(request, room_mode="shadow")
    with pytest.raises(ValueError, match="token"):
        plugin.score_highlight(request)
    assert request.rule_score == 0.77
    plugin.on_disable()
    with pytest.raises(RuntimeError):
        plugin.score_highlight(request)


def test_bundle_can_load_in_fresh_host_without_installed_plugin_package(host_source: Path, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/build_plugin.py")], capture_output=True, check=True)
    assert b"sha256" in completed.stdout
    import zipfile

    with zipfile.ZipFile(root / "dist/blc-inaspeech-0.1.0.zip") as archive:
        archive.extractall(tmp_path)
    code = """
import importlib.util, pathlib, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
spec = importlib.util.spec_from_file_location('entry', pathlib.Path(sys.argv[2])/'main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plugin = module.Plugin()
assert plugin.settings_schema
import blc_inaspeech
assert pathlib.Path(blc_inaspeech.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[2]).resolve())
assert 'tensorflow' not in sys.modules
assert 'inaSpeechSegmenter' not in sys.modules
print('bundle-only plugin import PASS')
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(host_source), str(tmp_path / "blc-inaspeech")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PASS" in process.stdout
