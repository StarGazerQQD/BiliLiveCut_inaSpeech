# BiliLiveCut inaSpeech 音频分类插件

独立的可选插件，通过 inaSpeechSegmenter 0.8.0 为录播片段生成 `speech`、`music`、`noise`、`silence` 时间区间和占比。

适用宿主：BiliLiveCut 0.1.19.0-alpha，插件 API 1。项目仓库：[StarGazerQQD/BiliLiveCut_inaSpeech](https://github.com/StarGazerQQD/BiliLiveCut_inaSpeech)。

当前通过宿主公开的 `highlight_scorer` 接口在转写后执行，使用 `shadow` 模式记录分类，保留宿主原评分；不会减少此前已执行的 ASR。它占用唯一高光插件槽位，不能与其他高光评分插件同时启用。不产生高光概率，不按“音乐”标签删除唱歌素材。

宿主适配器使用 Python 标准库。TensorFlow 和模型只存在于独立 Linux / Docker / WSL2 服务中，默认 CPU；CUDA 属于显式选择且须有可用 NVIDIA 环境。

## 完整说明

- [安装、Docker/WSL2、更新与排错](docs/INSTALLATION.md)
- [所有配置、时间语义及结果读取](docs/CONFIGURATION.md)
- [架构、HTTP 协议与失败处理](docs/ARCHITECTURE.md)
- [真实性能测量与后续升级评估](docs/PERFORMANCE.md)
- [安全检查、信任边界与维护](docs/SECURITY.md)
- [第三方来源和许可](docs/THIRD_PARTY.md)
- [本轮验收结果与环境限制](docs/VALIDATION.md)

本机 CPU 对 74.5 秒公开混合音频的最终测量中，优化配置中位耗时 1.32 秒，相对明确的重载模型/batch 32 基线降低约 44.4%；全部测量区间一致。此数值不含宿主解码和传输，且不是中文直播准确率或任意硬件加速保证，详见性能说明。

## 安装与启动

宿主需要可用的 FFmpeg。独立服务要求 Linux x86_64、Python 3.11–3.13；本轮真实验证使用 Docker 内 Python 3.12.15。Windows 宿主无需安装 TensorFlow，服务可放在 WSL2 或 Linux。Intel Arc 不能使用本项目的 CUDA 模式。

在服务所在 Linux 环境中进入本项目目录，使用 [uv](https://docs.astral.sh/uv/) 安装锁定依赖。若和 Windows 共用项目目录，使用单独的环境目录，不能共用 Windows `.venv`：

```bash
export UV_CACHE_DIR="$PWD/.local/uv-cache-linux"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-linux"
export TMPDIR="$PWD/.local/tmp"
mkdir -p "$TMPDIR"
uv sync --locked --extra runtime --no-dev
uv run --locked --extra runtime --no-dev blc-inaspeech token --output .local/inaspeech.token
uv run --locked --extra runtime --no-dev blc-inaspeech prepare-models --model-dir .local/models
uv run --locked --extra runtime --no-dev blc-inaspeech serve --model-dir .local/models --token-file .local/inaspeech.token
```

令牌命令拒绝覆盖已有文件，不在终端打印令牌。模型准备是唯一下载模型的步骤；服务启动时必须通过固定 SHA-256，正常推理不下载模型。官方模型资产未提供独立签名，这里的哈希是从固定官方 HTTPS 资产取得并随项目固定的内容身份。

默认监听 `127.0.0.1:8766`，串行推理、两个计算线程、batch 128、最多缓存 32 条结果。服务启动就绪后另开终端：

```bash
uv run --locked --extra runtime --no-dev blc-inaspeech health --token-file .local/inaspeech.token
uv run --locked --extra runtime --no-dev blc-inaspeech analyze /path/to/sample.mp4 --token-file .local/inaspeech.token
```

服务启动本身不需要 FFmpeg；`analyze` 命令及宿主适配器需要 FFmpeg。原始媒体由调用端解码，服务只接收标准 PCM WAV，不接受 URL 或调用端文件路径。

官方 0.8.0 的依赖声明包括 TensorFlow CUDA 组件和 ONNX Runtime GPU，即使用 CPU 也会安装它们。首次下载为 GB 级，磁盘占用更大；本项目遵循其依赖声明，不裁掉声明依赖、不改上游发行包。宿主插件压缩包不包含这些组件。

## 安装到 BiliLiveCut

1. 生成插件压缩包：在本项目开发环境运行 `python scripts/build_plugin.py`。
2. 将 `dist/blc-inaspeech-0.1.0.zip` 解压到宿主 `PLUGIN_DIR`，得到 `storage/plugins/blc-inaspeech/plugin.json`、`main.py` 和自有 `blc_inaspeech/` 包。不要只复制源码仓库中的两个入口文件。
3. 打开宿主插件中心，重新扫描，启用“inaSpeech 音频分类”，进入设置。
4. 填写推理服务地址、令牌文件中的 64 位字符串、宿主视角的授权媒体根目录及 FFmpeg 程序。密码字段空白保存沿用旧值，这是宿主已有行为。
5. 保存后分析一个新片段。分类结果记录到宿主插件观测及候选特征的 `highlight_plugin.prediction.metadata.audio_classification`；不是专属音频时间线界面。也可先通过上述 `analyze` 命令查看 JSON。

宿主只加载标准库客户端，不需要给宿主 Python 增装本项目或修改其依赖锁。正式压缩包包含所有必需自有文件。默认媒体根目录为空，必须由管理员明确配置；不允许目录外文件、符号链接媒体及非媒体后缀。

## 行为和限制

- 时间从送入模型的片段媒体起点开始，单位为秒，区间为 `[start_s, end_s)`。范围不是整场直播统计。
- `silence` 对应上游 `noEnergy`；不能把低能量判断当成准确的人声静音真值。
- 分类占比总和为 1，属于音频类别占比。插件没有高光训练集，不伪造 Champion/Shadow 高光概率，也不写训练反馈文件。
- 房间模式 `off` 跳过插件；其他模式仍只旁路记录，不覆盖宿主规则分。服务不可用、超时、音频或返回值非法时由宿主记录失败并保留原流程。
- 仅支持 0.72–600 秒、有音轨的媒体。更长、更短或损坏的片段明确失败，不会静默截断后伪装为完整分类。
- 不接受在线媒体 URL；宿主 FFmpeg 限制为本地协议和常见媒体解复用器，并移除来源元数据。
- 正常解码至多等待 30 秒，HTTP 至多等待 120 秒，可在设置中调整。单个宿主分析任务会等待插件；原生隔离不等于零资源竞争。
- 结果缓存按完整 WAV 内容哈希，无磁盘音频缓存；缓存有上限，服务重启失效。
- 当前宿主只支持一个高光评分插件；不要同时启用其他评分提供者。

## 安全与设备

服务令牌与宿主管理员密码相互独立。宿主密码字段隐藏回显，但宿主当前将插件设置保存在本地数据库，管理员需保护数据库及备份。不要提交 `.local`、令牌、环境、缓存或模型文件。

服务默认只监听环回地址，禁止 CORS、重定向和系统代理。远端服务使用 HTTPS；Docker 专用网络明文 HTTP 需要在插件中显式允许。不要将本地服务直接发布到公网。模型、超时和资源边界降低故障影响，但该插件不是安全沙箱。

CUDA 使用 `serve --device cuda`，只在有可用 NVIDIA CUDA 环境的 Linux/WSL2 中使用。启动时检测不到 GPU 会失败，不静默降到 CPU。服务启用显存按需增长，但与宿主 ASR 共享 GPU，仍可能争用显存和算力。本轮没有 NVIDIA 设备，CUDA 未实测。

## 开发与验收

开发使用 Python 3.12 和 uv。以下命令在项目目录运行，宿主源码参数必须指向真实 BiliLiveCut 源码；测试不修改它，也不会用跳过掩盖缺少宿主：

```text
uv sync --locked --group dev
uv run pytest tests/ --host-source /path/to/BiliLiveCut --basetemp .local/pytest
uv run ruff check src tests scripts plugin
uv run ruff format --check src tests scripts plugin
uv run mypy --no-incremental
uv build
uv run python scripts/build_plugin.py
```

真实模型另在装好运行依赖的 Linux 环境验收：

```text
python scripts/verify_runtime.py --model-dir .local/models --wav /path/to/16k-mono.wav --output .local/runtime-verification.json
```

本仓库不包含 BiliLiveCut 源码、第三方模型或媒体，测试用模型及公开音频存放在本地忽略目录。上游项目和模型来源：[ina-foss/inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)，其发行包按 MIT 许可分发。
