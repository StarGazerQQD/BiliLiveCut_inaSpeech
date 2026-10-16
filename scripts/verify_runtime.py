"""使用真实官方 CNN，验收分类、缓存及带鉴权 HTTP 往返。"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from blc_inaspeech.audio import wav_duration
from blc_inaspeech.client import Client
from blc_inaspeech.engine import Engine, load_predictor
from blc_inaspeech.server import InferenceServer


def main() -> None:
    """完整运行 CPU 模型，任一不满足立即以非零退出。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = load_predictor(args.model_dir)
    engine = Engine(predictor)
    data = args.wav.read_bytes()
    duration = wav_duration(data)
    token = "ab" * 32  # 仅用于本进程内、随机端口、生命周期有限的回归测试。
    with InferenceServer(("127.0.0.1", 0), engine, token) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            client = Client(f"http://127.0.0.1:{server.server_port}", token, timeout_s=300)
            health = client.health()
            first = client.analyze(data)
            cached = client.analyze(data)
            if first.cache_hit or not cached.cache_hit or first.segments != cached.segments:
                raise AssertionError("真实模型缓存结果不一致")
            if abs(sum(first.features().values()) - 1) > 1e-6 or first.duration_s != duration:
                raise AssertionError("真实模型时长或分类占比无效")
            args.output.write_text(
                json.dumps(
                    {
                        "health": health,
                        "first": first.to_dict(),
                        "cached": cached.to_dict(),
                        "features": first.features(),
                        "status": "passed",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print("真实 CPU 模型、HTTP 鉴权、协议和缓存验收通过。")
        finally:
            server.shutdown()
            thread.join(timeout=15)
            if thread.is_alive():
                raise AssertionError("验收服务未完成退出")


if __name__ == "__main__":
    main()
