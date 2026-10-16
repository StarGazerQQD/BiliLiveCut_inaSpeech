from __future__ import annotations

import http.client
import json
from collections.abc import Callable
from urllib.parse import urlsplit

import pytest

from blc_inaspeech.client import Client, validate_endpoint
from blc_inaspeech.engine import Engine, Predictor
from blc_inaspeech.schema import MAX_WAV_BYTES
from blc_inaspeech.server import Handler, InferenceServer

ServerFactory = Callable[[Predictor], tuple[str, str, Engine]]


def request(
    endpoint: str,
    token: str,
    *,
    method: str = "POST",
    path: str = "/v1/analyze",
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        values = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}
        values.update(headers or {})
        connection.request(method, path, body, values)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_real_http_roundtrip_and_cache(
    running_server: ServerFactory,
    predictor: Predictor,
    make_wav: Callable[..., bytes],
) -> None:
    endpoint, token, _ = running_server(predictor)
    client = Client(endpoint, token)
    assert client.health()["ready"] is True
    result = client.analyze(make_wav())
    assert result.features()["inaspeech_music_ratio"] == 0.5
    assert not result.cache_hit
    assert client.analyze(make_wav()).cache_hit


@pytest.mark.parametrize("token,headers", [("bad", {}), ("", {}), ("ab" * 32, {"Origin": "https://evil.test"})])
def test_authentication_and_cross_origin_rejection(
    running_server: ServerFactory,
    predictor: Predictor,
    token: str,
    headers: dict[str, str],
) -> None:
    endpoint, _, _ = running_server(predictor)
    status, body = request(endpoint, token, headers=headers)
    assert status == 401 and b"ab" * 32 not in body


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Content-Length": str(MAX_WAV_BYTES + 1)}, 413),
        ({"Content-Length": "-1"}, 400),
        ({"Transfer-Encoding": "chunked"}, 400),
        ({"Content-Type": "application/json"}, 415),
    ],
)
def test_request_framing_limits(
    running_server: ServerFactory,
    predictor: Predictor,
    headers: dict[str, str],
    status: int,
) -> None:
    endpoint, token, _ = running_server(predictor)
    assert request(endpoint, token, body=b"x" * 44, headers=headers)[0] == status


def test_bad_audio_routes_and_model_output_isolation(
    running_server: ServerFactory,
    predictor: Predictor,
    make_wav: Callable[..., bytes],
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint, token, _ = running_server(predictor)
    assert request(endpoint, token, body=b"x" * 44)[0] == 422
    assert request(endpoint, token, path="/v1/analyze?file=/etc/passwd")[0] == 404
    assert request(endpoint, token, method="GET", path="/v1/health?query=1")[0] == 404

    def broken(path: str) -> list[tuple[str, float, float]]:
        raise RuntimeError("secret-token-and-private-path")

    broken_endpoint, _, _ = running_server(broken)
    status, body = request(broken_endpoint, token, body=make_wav())
    assert status == 503 and b"secret" not in body
    assert any(record.exc_info is not None and "inference_unavailable" in record.message for record in caplog.records)
    with pytest.raises(RuntimeError, match="503"):
        Client(broken_endpoint, token).analyze(make_wav())


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/socket",
        "http://localhost:8766",
        "http://example.com",
        "http://127.0.0.1/admin",
        "http://user:secret@127.0.0.1",
        "http://127.0.0.1?url=evil",
        "http://127.0.0.1#fragment",
        "http://127.0.0.1:0",
        "http://127.0.0.1\n",
        "http://127.0.0.1\\evil",
    ],
)
def test_endpoint_rejects_credentials_paths_dns_loopback_and_unsafe_network_http(url: str) -> None:
    with pytest.raises(ValueError):
        validate_endpoint(url)


def test_explicit_network_http_and_https_allowed() -> None:
    assert validate_endpoint("http://ina-service:8766", allow_network_http=True).host == "ina-service"
    assert validate_endpoint("https://audio.example.test").tls
    assert validate_endpoint("http://[::1]:8766").port == 8766


def test_duplicate_headers_rejected(running_server: ServerFactory, predictor: Predictor) -> None:
    endpoint, token, _ = running_server(predictor)
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        connection.putrequest("POST", "/v1/analyze")
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Length", "44")
        connection.putheader("Content-Length", "44")
        connection.putheader("Content-Type", "audio/wav")
        connection.endheaders(b"x" * 44)
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read())["error"] == "invalid_framing"
    finally:
        connection.close()


@pytest.mark.parametrize(
    "status,body",
    [
        (302, b"{}"),
        (200, b'{"ready":true,"ready":false}'),
        (200, b'{"ready":NaN}'),
        (200, b"{}"),
    ],
)
def test_client_rejects_redirect_and_invalid_health(
    running_server: ServerFactory,
    predictor: Predictor,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: bytes,
) -> None:
    def respond(handler: Handler) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(Handler, "do_GET", respond)
    endpoint, token, _ = running_server(predictor)
    with pytest.raises((ValueError, RuntimeError)):
        Client(endpoint, token).health()


def test_slow_request_body_has_total_deadline(
    running_server: ServerFactory,
    predictor: Predictor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(InferenceServer, "body_timeout_s", 0.05)
    endpoint, token, _ = running_server(predictor)
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        connection.putrequest("POST", "/v1/analyze")
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", "audio/wav")
        connection.putheader("Content-Length", "64")
        connection.endheaders(b"x")
        response = connection.getresponse()
        assert response.status == 408
    finally:
        connection.close()
