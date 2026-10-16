"""生成包含自有轻量包的可复制插件压缩包及真实文件校验清单。"""

from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """按固定时间与稳定顺序构建，不包含密钥、环境、模型或宿主源码。"""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    manifest = json.loads((ROOT / "plugin/blc-inaspeech/plugin.json").read_text(encoding="utf-8"))
    if manifest["version"] != project["version"]:
        raise ValueError("插件版本与项目版本不一致")
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / f"blc-inaspeech-{project['version']}.zip"
    sources = [(p, f"blc-inaspeech/{p.name}") for p in (ROOT / "plugin/blc-inaspeech").iterdir() if p.is_file()]
    package = ROOT / "src/blc_inaspeech"
    sources.extend(
        (p, f"blc-inaspeech/blc_inaspeech/{p.relative_to(package).as_posix()}")
        for p in package.rglob("*")
        if p.is_file() and p.suffix in {".py", ".json"}
    )
    sources.append((ROOT / "LICENSE", "blc-inaspeech/LICENSE"))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as writer:
        for source, destination in sorted(sources, key=lambda item: item[1]):
            info = zipfile.ZipInfo(destination, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            writer.writestr(info, source.read_bytes())
    data = archive.read_bytes()
    metadata = {
        "file": archive.name,
        "version": project["version"],
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "crc32": f"{zlib.crc32(data):08x}",
    }
    (output / "plugin-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    artifacts: list[dict[str, object]] = []
    for artifact in sorted(output.iterdir()):
        if artifact.is_file() and artifact.suffix in {".zip", ".whl", ".gz"}:
            content = artifact.read_bytes()
            artifacts.append(
                {
                    "file": artifact.name,
                    "version": project["version"],
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "crc32": f"{zlib.crc32(content):08x}",
                }
            )
    (output / "artifact-manifest.json").write_text(
        json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
