# 安装、更新与排错

## 验证范围

宿主公共接口以 BiliLiveCut `0.1.19.0-alpha`、API `1` 为准，测试快照为已合并的 `f2a5df0`，对应 main 合并提交 `285e04dcc90477f6d395f8392272d3f52858d1d6` 的插件契约。

| 部署端 | 支持情况 | 本轮证据 |
| --- | --- | --- |
| Windows 宿主客户端 | 支持 Python 3.11–3.13，推荐 3.12 | Windows 3.12.14、真实 FFmpeg、真实容器服务往返 |
| Linux 独立服务 | 支持 x86_64，Python 3.11–3.13，推荐 3.12 | Python 3.12.15 真实 CNN 和完整测试 |
| Docker Compose CPU | 支持 Linux 容器 | 固定官方镜像、非 root、环回端口和健康检查均实际验证；镜像 Python 3.12.14 |
| Windows 原生 TensorFlow 服务 | 不作为本项目的安装路径 | 请使用 Docker Desktop 或 WSL2 中的 Linux |
| WSL2 直接部署 | 按 Linux 路径安装 | 本轮使用 WSL2 后端的 Docker，没有单独验收 Ubuntu 发行版 |
| NVIDIA CUDA | 提供显式配置，未实测 | 测试机器为 Intel Arc，不具备 NVIDIA CUDA |
| macOS / ARM64 | 未提供服务验收 | 不承诺官方模型依赖的跨平台安装 |

## 推荐：Docker Desktop / Docker Engine

需要已就绪的 Linux 容器引擎及 Docker Compose。依赖、模型、日志和令牌都保存在项目 `.local/` 中，不会装进 BiliLiveCut 的 Python。

首先在项目目录创建准备和运行临时目录。Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force .local/tmp,.local/runtime-data/tmp
```

Linux/WSL 使用 `mkdir -p .local/tmp .local/runtime-data/tmp`。若项目在 Linux 原生文件系统，复制 `.env.example` 为 `.env`，把 `INASPEECH_UID/GID` 改为 `id -u` / `id -g` 的值，确保普通用户拥有 `.local`；不使用 root。

准备依赖清单需要本机安装 [uv](https://docs.astral.sh/uv/)。源码中的 `uv.lock` 是依赖事实来源，导出时保留全部哈希：

```text
uv export --locked --extra runtime --no-dev --no-emit-project --format requirements-txt --output-file .local/runtime-requirements.txt
docker compose pull
docker compose -f compose.yaml -f compose.setup.yaml run --rm --no-deps --entrypoint uv inaspeech venv --python /usr/local/bin/python .local/runtime-venv
docker compose -f compose.yaml -f compose.setup.yaml run --rm --no-deps --entrypoint uv inaspeech pip sync --python .local/runtime-venv/bin/python --require-hashes --only-binary :all: -r .local/runtime-requirements.txt
docker compose -f compose.yaml -f compose.setup.yaml run --rm --no-deps --entrypoint /usr/local/bin/python inaspeech -m blc_inaspeech token --output .local/inaspeech.token
docker compose -f compose.yaml -f compose.setup.yaml run --rm --no-deps --entrypoint /usr/local/bin/python inaspeech -m blc_inaspeech prepare-models --model-dir .local/models
docker compose up --detach --wait --wait-timeout 180
docker compose ps
```

已有有效运行环境时不重复执行 `venv`。已有令牌时不重复执行 `token`；该命令会拒绝覆盖，升级不需要更换服务令牌。

Compose 使用固定摘要的官方 `ghcr.io/astral-sh/uv` 镜像，不需要自行构建镜像。正常服务中源码、依赖、令牌及模型均只读，只有 `.local/runtime-data` 可写；根文件系统只读，删除全部 Linux capabilities，并设置 CPU、内存和进程数限制。`compose.setup.yaml` 仅为显式安装步骤开放 `.local` 写权限，正常启动不要加载它。镜像及引擎自身的数据由 Docker 管理，项目文件和运行数据均位于本目录。

容器内服务监听 `0.0.0.0`，发布到 Windows/Linux 主机的端口仅为 `127.0.0.1:8766`。宿主插件地址填写 `http://127.0.0.1:8766`。

停止与查看日志：

```text
docker compose logs --tail 50 inaspeech
docker compose down
```

`down` 不删除 `.local` 中的依赖、模型、令牌或缓存目录。正常停止通过 SIGTERM 展开资源上下文，释放请求临时音频。强制终止或断电可能留下 `.local/runtime-data/tmp/blc-ina-*`（直接运行时在 TMPDIR 下）的临时目录；先停止服务，再由管理员核对清理。

## WSL2 / Linux 直接运行

需要 Linux x86_64、Python 3.12 和 uv；完整命令见 [README](../README.md#安装与启动)。直接运行时的环境使用 `.venv-linux`，不要与 Windows `.venv` 或 Docker `.local/runtime-venv` 交叉同步。

独立服务本身不依赖 FFmpeg；Windows 宿主已负责解码。只有在 Linux 使用 `analyze` CLI 处理压缩媒体时，Linux 也需要安装 FFmpeg。

WSL2 的模型和依赖放在 Linux 文件系统通常更利于启动及大量小文件访问。本聊天交付的所有文件均留在指定 Windows 项目目录，未主动迁移到其他位置。若用户自行迁移，仍须保持模型哈希和服务令牌一致。

WSL2 网络转发因系统设置而异。Windows 能访问的服务地址应先用 `health` 实际验证；不要以关闭认证或把端口开放到公网解决连接问题。参考 [微软 WSL 网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking)。

## 宿主安装

开发环境执行 `uv sync --locked --group dev`，随后 `uv run python scripts/build_plugin.py`。将 `dist/blc-inaspeech-0.1.0.zip` 解压到宿主 `PLUGIN_DIR`，保持目录名 `blc-inaspeech`。ZIP 已包含自有客户端包，宿主不需要额外安装本项目。

在宿主插件中心重新扫描并启用，进入设置，填写：

1. 服务地址：默认 `http://127.0.0.1:8766`。
2. 令牌：`.local/inaspeech.token` 的内容，不是文件路径。
3. 授权媒体根目录：必须是宿主视角可见的录制或导入素材目录；不要填写服务容器的路径。
4. FFmpeg：宿主已有可执行文件的路径或 `ffmpeg`。

独立服务只能看见上传的音频字节，不需要挂载宿主录制目录。若 BiliLiveCut 本身也在容器内，两者通过受信任专用网络连接时，可显式启用非环回 HTTP；否则使用 HTTPS。当前 Compose 不自动加入其他项目网络，也不自动改变宿主配置。

当前宿主只允许一个高光评分插件。本插件不是其他评分模型的自动组合器，启用冲突由宿主拒绝。

## CUDA

需要 NVIDIA 显卡、兼容的 Windows/Linux 驱动，以及已确认可用的 Linux/WSL2 CUDA 容器环境。Intel Arc 不能使用此配置。

将 `.env` 中 `INASPEECH_DEVICE` 设为 `cuda`，然后：

```text
docker compose -f compose.yaml -f compose.gpu.yaml up --detach --wait --wait-timeout 180
```

服务检测不到 GPU 会直接失败，不将 CPU 结果冒充 CUDA。模型使用显存按需增长，仍会与 ASR 争用显存和算力。batch、并发和配额必须结合显卡实测；本轮没有 CUDA 性能或兼容性通过结论。

## 更新、回滚及离线安装

更新前停用宿主插件、停止独立服务，再更换源码及插件 ZIP，按新锁文件同步独立环境。令牌和宿主数据库不需要删除；只有模型清单变化时才重新准备模型。当前进程已加载的旧代码不会自动热更新。

回滚使用上一版源码、锁文件、模型清单和 ZIP，不只更换入口文件。插件本身没有数据库迁移或训练文件。

离线机器可以复制已验证模型文件及对应 `models.json`。依赖需提前从锁定清单准备目标 Linux/Python ABI 的完整官方 wheels，使用 uv 的 `--offline --no-index --find-links` 并保留 `--require-hashes`；不能复制 Windows 虚拟环境给 Linux。

官方 0.8.0 声明了 GPU 依赖，首次下载为 GB 级。常驻运行避免反复导入，不能消除首次安装成本。不要为压缩安装包随意裁掉声明依赖或使用来历不明的模型。

## 常见故障

| 表现 | 检查和处理 |
| --- | --- |
| 插件扫描不到 | 目录名、`plugin.json` 和完整 ZIP 层级是否正确 |
| 提示已有高光提供者 | 先停用另一个评分插件；当前接口不能同时启用 |
| HTTP 401 | 令牌是否与服务文件一致；密码空白保存会保留旧值 |
| HTTP 422 | 输入 WAV 或模型输出无效，检查是否有音轨及 0.72–600 秒长度限制 |
| HTTP 408 | 上传未在 10 秒内完成；检查本机或专用网络，不扩大到公网 |
| HTTP 503 | 查看服务日志，保留宿主回退结果后定位模型或文件系统错误 |
| 模型哈希失败 | 停止服务，核对来源和固定清单；不改清单迎合损坏文件 |
| 首次启动较慢 | 首次导入 TensorFlow 及跨系统文件访问较慢，等待健康检查 |
| CPU 模式打印 `cuInit` 失败 | TensorFlow GPU 发行包的探测日志；结合 `--device cpu` 和实际成功结果判断，不将其视为 CUDA 已启用 |
| Windows 全量 pytest 卡在 `socketpair.accept` | 本轮记录的标准库环境问题；Linux 完整测试和 Windows 专项分别验收，不删除测试或修改宿主事件循环 |
