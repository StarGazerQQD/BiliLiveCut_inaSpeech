from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import Request

import pytest

from blc_inaspeech import cli, models


class Response(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        # 固定长度已经足够验证模型；禁止额外读一字节并等待远端 EOF。
        remaining = len(self.getvalue()) - self.tell()
        assert 0 < size <= remaining
        return super().read(size)


class Opener:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def open(self, url: str, timeout: int) -> Response:
        assert url.startswith("https://github.com/") and timeout == 60
        return Response(self.data)


def test_model_download_exact_length_atomic_write_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"trusted fixture model"
    name = "keras_speech_music_noise_cnn.hdf5"
    spec = models.Model(name, "https://github.com/ina-foss/model", hashlib.sha256(data).hexdigest(), len(data))
    monkeypatch.setattr(models, "model_manifest", lambda: {name: spec})
    monkeypatch.setattr(models.urllib.request, "build_opener", lambda *handlers: Opener(data))
    assert models.prepare_models(tmp_path) == [(tmp_path / name).resolve()]
    assert (tmp_path / name).read_bytes() == data
    assert not list(tmp_path.glob("*.part"))
    assert models.prepare_models(tmp_path) == [(tmp_path / name).resolve()]


def test_tampered_download_leaves_no_destination_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "keras_speech_music_noise_cnn.hdf5"
    spec = models.Model(name, "https://github.com/ina-foss/model", "0" * 64, 5)
    monkeypatch.setattr(models, "model_manifest", lambda: {name: spec})
    monkeypatch.setattr(models.urllib.request, "build_opener", lambda *handlers: Opener(b"wrong"))
    with pytest.raises(ValueError, match="身份不匹配"):
        models.prepare_models(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["http://github.com/model", "https://evil.test/model", "file:///secret"])
def test_model_download_rejects_untrusted_redirect(target: str) -> None:
    with pytest.raises(ValueError, match="重定向"):
        models._HTTPSRedirect().redirect_request(Request("https://github.com/model"), None, 302, "", {}, target)


def test_cli_token_does_not_print_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "service.token"
    monkeypatch.setattr("sys.argv", ["blc-inaspeech", "token", "--output", str(path)])
    cli.main()
    token = cli.read_token(path)
    assert token not in capsys.readouterr().out
    assert path.stat().st_size == 65


def test_cli_prepare_only_calls_explicit_install_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["blc-inaspeech", "prepare-models", "--model-dir", str(tmp_path)])

    def prepare(directory: Path, *, include_gender: bool) -> list[Path]:
        assert directory == tmp_path and not include_gender
        return [directory / "fixture.hdf5"]

    monkeypatch.setattr(cli, "prepare_models", prepare)
    cli.main()
    assert "fixture.hdf5" in capsys.readouterr().out


def test_sigterm_handler_unwinds_resource_contexts(tmp_path: Path) -> None:
    import tempfile

    temporary: Path | None = None
    with pytest.raises(KeyboardInterrupt):
        with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
            temporary = Path(directory)
            (temporary / "audio.wav").write_bytes(b"private fixture")
            cli._interrupt(15, None)
    assert temporary is not None and not temporary.exists()
