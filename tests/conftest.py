from __future__ import annotations

import io
import math
import os
import struct
import threading
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from blc_inaspeech.engine import Engine, Predictor
from blc_inaspeech.server import InferenceServer


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--host-source", default=os.environ.get("BLC_HOST_SOURCE"), help="真实 BiliLiveCut 源码目录")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None and terminal.stats.get("skipped"):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def host_source(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    value = request.config.getoption("--host-source")
    if not value:
        pytest.fail("完整宿主联调需指定 --host-source <BiliLiveCut目录>，禁止以跳过代替验收")
    source = Path(value).resolve(strict=True)
    if not (source / "app/plugins/contracts.py").is_file():
        pytest.fail("--host-source 不包含真实插件契约")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'host.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("PLUGIN_DIR", str(tmp_path / "plugins"))
    monkeypatch.syspath_prepend(str(source))
    from app.core.config import get_settings

    get_settings.cache_clear()
    assert get_settings().database_url == os.environ["DATABASE_URL"]
    return source


@pytest.fixture
def make_wav() -> Callable[..., bytes]:
    def build(duration: float = 1.0, *, rate: int = 16000, channels: int = 1, frequency: float = 440.0) -> bytes:
        stream = io.BytesIO()
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            frames = int(duration * rate)
            pcm = b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * index / rate))) * channels
                for index in range(frames)
            )
            writer.writeframes(pcm)
        return stream.getvalue()

    return build


@pytest.fixture
def running_server() -> Iterator[Callable[[Predictor], tuple[str, str, Engine]]]:
    servers: list[tuple[InferenceServer, threading.Thread]] = []

    def start(predictor: Predictor) -> tuple[str, str, Engine]:
        token = "ab" * 32
        engine = Engine(predictor, cache_entries=2)
        server = InferenceServer(("127.0.0.1", 0), engine, token)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", token, engine

    yield start
    for server, thread in reversed(servers):
        server.shutdown()
        server.server_close()
        thread.join(timeout=15)
        assert not thread.is_alive()


@pytest.fixture
def predictor() -> Predictor:
    def predict(path: str) -> list[tuple[str, float, float]]:
        with wave.open(path, "rb") as reader:
            duration = reader.getnframes() / reader.getframerate()
        return [("speech", 0.0, duration / 2), ("music", duration / 2, duration)]

    return predict
