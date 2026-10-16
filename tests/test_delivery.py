from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_all_relative_document_links_resolve() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for document in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            assert (document.parent / relative).resolve().is_file(), (document.name, target)


def test_default_service_keeps_source_secrets_and_dependencies_readonly() -> None:
    config = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = config["services"]["inaspeech"]
    assert "@sha256:" in service["image"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["ports"] == ["127.0.0.1:8766:8766"]
    assert service["user"] == "${INASPEECH_UID:-1000}:${INASPEECH_GID:-1000}"
    assert service["volumes"] == [
        ".:/work:ro",
        "./.local:/work/.local:ro",
        "./.local/runtime-data:/work/.local/runtime-data:rw",
    ]
    assert service["pids_limit"] == 128
    assert "pip" not in service["command"] and "prepare-models" not in service["command"]
    assert "gpus" not in service
    setup = yaml.safe_load((ROOT / "compose.setup.yaml").read_text(encoding="utf-8"))
    assert setup["services"]["inaspeech"]["volumes"] == ["./.local:/work/.local:rw"]


def test_public_manifest_versions_and_dependency_sources_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    plugin = json.loads((ROOT / "plugin/blc-inaspeech/plugin.json").read_text(encoding="utf-8"))
    assert project["project"]["version"] == plugin["version"] == "0.1.0"
    assert plugin["api_version"] == "1" and plugin["capabilities"] == ["highlight_scorer"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        source = package["source"]
        assert source == {"editable": "."} or source == {"registry": "https://pypi.org/simple"}
        for wheel in package.get("wheels", []):
            assert wheel["url"].startswith("https://files.pythonhosted.org/")
            assert wheel["hash"].startswith("sha256:")
