"""不读取系统代理、无重定向、带固定响应上限的服务客户端。"""

from __future__ import annotations

import http.client
import ipaddress
import json
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit

from blc_inaspeech.audio import wav_duration
from blc_inaspeech.models import model_manifest
from blc_inaspeech.schema import MAX_RESPONSE_BYTES, Analysis


@dataclass(frozen=True, slots=True)
class Endpoint:
    """已验证的服务地址。"""

    host: str
    port: int
    tls: bool


def validate_endpoint(url: str, *, allow_network_http: bool = False) -> Endpoint:
    """默认只接受环回 HTTP；远程 HTTPS 显式使用证书校验。"""
    if any(ord(character) <= 32 or ord(character) == 127 for character in url) or "\\" in url:
        raise ValueError("服务地址包含空白或非法字符")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("服务地址必须为无路径、无凭据的 HTTP/HTTPS origin")
    host = parsed.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False  # 不把可被 DNS/hosts 重定向的 localhost 当作已验证环回。
    if parsed.scheme == "http" and not loopback and not allow_network_http:
        raise ValueError("非环回 HTTP 默认禁止；Docker 专用网络须显式允许")
    if parsed.port == 0:
        raise ValueError("服务端口不能为零")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return Endpoint(host, port, parsed.scheme == "https")


class Client:
    """每次调用释放连接；HTTP 错误或协议错误明确交由宿主回退。"""

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        timeout_s: float = 120.0,
        allow_network_http: bool = False,
    ) -> None:
        self.endpoint = validate_endpoint(endpoint, allow_network_http=allow_network_http)
        if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
            raise ValueError("服务令牌必须是 64 位小写十六进制随机字符串")
        if not 1 <= timeout_s <= 300:
            raise ValueError("推理请求超时必须为 1 至 300 秒")
        self.token = token
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, body: bytes | None = None) -> object:
        endpoint = self.endpoint
        connection: http.client.HTTPConnection
        if endpoint.tls:
            connection = http.client.HTTPSConnection(
                endpoint.host,
                endpoint.port,
                timeout=self.timeout_s,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=self.timeout_s)
        try:
            connection.request(
                method,
                path,
                body=body,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "audio/wav", "Connection": "close"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"inaSpeech 服务拒绝请求（HTTP {response.status}）")
            if response.getheader("Content-Type") != "application/json":
                raise ValueError("服务未返回 JSON")
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("推理服务响应过大")
            return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        finally:
            connection.close()

    def health(self) -> dict[str, object]:
        """检查带鉴权的就绪接口。"""
        result = self._request("GET", "/v1/health")
        model = model_manifest()["keras_speech_music_noise_cnn.hdf5"]
        if not isinstance(result, dict) or result != {
            "ready": True,
            "schema_version": "1",
            "model_sha256": model.sha256,
        }:
            raise ValueError("服务未就绪或模型身份不符")
        return result

    def analyze(self, wav: bytes) -> Analysis:
        """校验输入、固定模型身份及输出覆盖时长。"""
        duration = wav_duration(wav)
        model = model_manifest()["keras_speech_music_noise_cnn.hdf5"]
        result = Analysis.from_dict(self._request("POST", "/v1/analyze", wav), expected_model=model.sha256)
        if abs(result.duration_s - duration) > 1 / 16000:
            raise ValueError("服务结果与上传音频时长不符")
        return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"JSON 禁止非有限常量：{value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复字段")
        result[key] = value
    return result
