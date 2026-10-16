"""默认环回、强令牌鉴权、串行推理、限长且不提供文件路径 API 的 HTTP 服务。"""

from __future__ import annotations

import hmac
import json
import logging
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast

from blc_inaspeech.engine import Engine
from blc_inaspeech.schema import MAX_WAV_BYTES


class InferenceServer(HTTPServer):
    """同一时间只有一个模型调用，避免显存竞争和模型线程不安全。"""

    request_queue_size = 4
    body_timeout_s = 10.0

    def __init__(self, address: tuple[str, int], engine: Engine, token: str) -> None:
        if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("服务令牌必须是 64 位小写十六进制随机字符串")
        self.engine = engine
        self.token = token
        super().__init__(address, Handler)

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        """每条连接都有有限读写等待时间。"""
        stream, address = super().get_request()
        stream.settimeout(10.0)
        return stream, address


class Handler(BaseHTTPRequestHandler):
    """只接受两个固定路由，错误响应不回显路径、音频或令牌。"""

    server_version = "blc-inaspeech"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        """不记录请求行、访问令牌和媒体内容。"""

    def _send(self, status: int, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _authorized(self) -> bool:
        server = cast(InferenceServer, self.server)
        authorization = self.headers.get_all("Authorization", [])
        # 不提供 CORS；浏览器跨站请求与来源不明的请求体均不能触发推理。
        if (
            self.headers.get("Origin") is not None
            or len(authorization) != 1
            or not hmac.compare_digest(authorization[0].encode("utf-8"), f"Bearer {server.token}".encode("ascii"))
        ):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        """健康检查只有在模型已初始化且请求通过鉴权后可用。"""
        if not self._authorized():
            return
        if self.path != "/v1/health":
            self._send(404, {"error": "not_found"})
            return
        server = cast(InferenceServer, self.server)
        self._send(200, {"ready": True, "schema_version": "1", "model_sha256": server.engine.model_hash})

    def do_POST(self) -> None:
        """接收规范 WAV，不接受 URL、主机文件路径或任意执行参数。"""
        if not self._authorized():
            return
        if self.path != "/v1/analyze":
            self._send(404, {"error": "not_found"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            self._send(400, {"error": "invalid_framing"})
            return
        try:
            if not lengths[0].isascii() or not lengths[0].isdigit():
                raise ValueError("length")
            length = int(lengths[0])
        except ValueError:
            self._send(400, {"error": "invalid_length"})
            return
        if not 44 <= length <= MAX_WAV_BYTES:
            self._send(413, {"error": "body_too_large_or_small"})
            return
        if self.headers.get_all("Content-Type", []) != ["audio/wav"]:
            self._send(415, {"error": "wav_required"})
            return
        try:
            chunks: list[bytes] = []
            remaining = length
            server = cast(InferenceServer, self.server)
            deadline = time.monotonic() + server.body_timeout_s
            while remaining:
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError("body deadline exceeded")
                self.connection.settimeout(seconds)
                chunk = self.rfile.read1(min(remaining, 65536))
                if not chunk:
                    raise ValueError("incomplete")
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            analysis = server.engine.analyze(data)
        except TimeoutError:
            self._send(408, {"error": "body_timeout"})
            return
        except ValueError:
            self._send(422, {"error": "invalid_audio_or_model_output"})
            return
        except (OSError, RuntimeError):
            logging.getLogger("blc_inaspeech.server").exception(
                "inference_unavailable: inspect protected service logs; response omits internal details"
            )
            self._send(503, {"error": "inference_unavailable"})
            return
        self._send(200, analysis.to_dict())
