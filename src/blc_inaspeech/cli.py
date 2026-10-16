"""模型准备、令牌初始化、服务启动和单文件测试入口。"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
from pathlib import Path
from types import FrameType
from typing import NoReturn

from blc_inaspeech.audio import decode_media
from blc_inaspeech.client import Client
from blc_inaspeech.engine import Engine, load_predictor
from blc_inaspeech.models import prepare_models
from blc_inaspeech.server import InferenceServer


def _interrupt(_signum: int, _frame: FrameType | None) -> NoReturn:
    """让容器 SIGTERM 走正常栈展开，释放请求临时目录和监听端口。"""
    raise KeyboardInterrupt


def read_token(path: Path) -> str:
    """只接受小型普通令牌文件，不允许链接到其他文件。"""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128:
        raise ValueError("令牌文件不存在、不是普通文件或过大")
    token = path.read_text(encoding="ascii").strip()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("令牌文件格式无效")
    return token


def create_token(path: Path) -> None:
    """生成独立服务令牌，拒绝覆盖，不在标准输出中显示秘密。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
        stream.write(secrets.token_hex(32) + "\n")


def parser() -> argparse.ArgumentParser:
    """构造所有显式操作，默认不启动服务。"""
    root = argparse.ArgumentParser(description="BiliLiveCut inaSpeech 独立分类服务")
    actions = root.add_subparsers(dest="command", required=True)
    token = actions.add_parser("token", help="生成随机服务令牌，拒绝覆盖")
    token.add_argument("--output", type=Path, default=Path(".local/inaspeech.token"))
    models = actions.add_parser("prepare-models", help="显式下载官方模型并验证 SHA-256")
    models.add_argument("--model-dir", type=Path, default=Path(".local/models"))
    models.add_argument("--include-gender", action="store_true", help="仅供对比基线，不在生产服务中启用")
    serve = actions.add_parser("serve", help="启动常驻服务；先准备模型和令牌")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--model-dir", type=Path, default=Path(".local/models"))
    serve.add_argument("--token-file", type=Path, default=Path(".local/inaspeech.token"))
    serve.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    serve.add_argument("--batch-size", type=int, default=128)
    serve.add_argument("--threads", type=int, default=2)
    serve.add_argument("--cache-entries", type=int, default=32)
    for action in (actions.add_parser("health"), actions.add_parser("analyze")):
        action.add_argument("--endpoint", default="http://127.0.0.1:8766")
        action.add_argument("--token-file", type=Path, default=Path(".local/inaspeech.token"))
        action.add_argument("--timeout", type=float, default=120)
    analyze = actions.choices["analyze"]
    analyze.add_argument("media", type=Path)
    analyze.add_argument("--ffmpeg", default="ffmpeg")
    return root


def main() -> None:
    """执行一个前台操作，错误保持非零退出及原始异常链。"""
    arguments = parser().parse_args()
    if arguments.command == "token":
        create_token(arguments.output)
        print("服务令牌已创建；请在宿主插件的密码设置中填写该文件内容。")
    elif arguments.command == "prepare-models":
        for path in prepare_models(arguments.model_dir, include_gender=arguments.include_gender):
            print(f"模型校验通过：{path.name}")
    elif arguments.command == "serve":
        token = read_token(arguments.token_file)
        predictor = load_predictor(
            arguments.model_dir,
            batch_size=arguments.batch_size,
            device=arguments.device,
            threads=arguments.threads,
        )
        engine = Engine(predictor, cache_entries=arguments.cache_entries)
        with InferenceServer((arguments.host, arguments.port), engine, token) as server:
            previous = signal.signal(signal.SIGTERM, _interrupt)
            print("inaSpeech 模型就绪；服务已启动。", flush=True)
            try:
                server.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                print("inaSpeech 服务已停止。")
            finally:
                signal.signal(signal.SIGTERM, previous)
    else:
        client = Client(arguments.endpoint, read_token(arguments.token_file), timeout_s=arguments.timeout)
        payload = (
            client.health()
            if arguments.command == "health"
            else client.analyze(decode_media(arguments.media, ffmpeg=arguments.ffmpeg)).to_dict()
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
