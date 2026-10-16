"""在真实服务容器中验证非 root 身份和文件系统最小写权限。"""

from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path


def main() -> None:
    """只在自有路径创建独占探针，清理后输出无秘密的验收结果。"""
    if sys.platform != "linux":
        raise RuntimeError("此验收必须在 Linux 服务容器中执行")
    uid = os.getuid()
    if uid == 0:
        raise AssertionError("服务不能以 root 运行")
    denied: list[str] = []
    for root in ("/work/src", "/work/.local/models", "/work/.local/runtime-venv"):
        directory = Path(root)
        if not directory.is_dir():
            raise AssertionError("必须先准备完整运行目录")
        probe = directory / "__blc_permission_probe__"
        try:
            with probe.open("xb"):
                pass
        except OSError as exc:
            if exc.errno not in {errno.EROFS, errno.EACCES, errno.EPERM}:
                raise
            denied.append(root)
        else:
            probe.unlink()
            raise AssertionError(f"目录不应可写：{root}")
    writable = Path("/work/.local/runtime-data/__blc_permission_probe__")
    try:
        with writable.open("x", encoding="utf-8") as stream:
            stream.write("permission fixture")
    finally:
        writable.unlink(missing_ok=True)
    print(json.dumps({"uid": uid, "read_only_verified": denied, "runtime_data_writable": True}))


if __name__ == "__main__":
    main()
