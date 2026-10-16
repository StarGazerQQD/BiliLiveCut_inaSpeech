"""显式下载并验证官方模型；正常推理路径不包含网络下载。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Model:
    """固定的官方发行模型身份。"""

    name: str
    url: str
    sha256: str
    size: int


def model_manifest() -> dict[str, Model]:
    """读取随软件分发的固定模型清单。"""
    raw = json.loads(files("blc_inaspeech").joinpath("models.json").read_text(encoding="utf-8"))
    return {name: Model(name, **data) for name, data in raw["models"].items()}


def verify_model(directory: Path, name: str) -> Path:
    """只接受清单中的普通文件，并逐次验证真实字节身份。"""
    spec = model_manifest().get(name)
    if spec is None:
        raise ValueError("模型未登记")
    target = directory / name
    if target.is_symlink() or not target.is_file() or target.stat().st_size != spec.size:
        raise ValueError(f"模型不存在、不是普通文件或大小不符：{name}")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"模型 SHA-256 不符：{name}")
    return target.resolve()


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        """只允许 GitHub 模型下载重定向到官方 HTTPS 资产域名。"""
        from urllib.parse import urlsplit

        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com"}:
            raise ValueError("拒绝非官方模型下载重定向")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def prepare_models(directory: Path, *, include_gender: bool = False) -> list[Path]:
    """由用户显式调用，限长下载到临时文件，校验后原子写入。"""
    directory.mkdir(parents=True, exist_ok=True)
    selected = ["keras_speech_music_noise_cnn.hdf5"]
    if include_gender:
        selected.append("keras_male_female_cnn.hdf5")
    result: list[Path] = []
    # 安装步骤允许管理员已有的 HTTPS 代理；推理客户端不读取系统代理。
    opener = urllib.request.build_opener(_HTTPSRedirect())
    for name in selected:
        target = directory / name
        if target.exists() or target.is_symlink():
            result.append(verify_model(directory, name))
            continue
        spec = model_manifest()[name]
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".model-", suffix=".part")
        try:
            with os.fdopen(descriptor, "wb") as output, opener.open(spec.url, timeout=60) as response:
                chunks: list[bytes] = []
                remaining = spec.size
                deadline = time.monotonic() + 120
                while remaining:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("官方模型下载超过总时限")
                    chunk = response.read(min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
                    raise ValueError(f"官方模型下载身份不匹配：{name}")
                output.write(data)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        result.append(verify_model(directory, name))
    return result
