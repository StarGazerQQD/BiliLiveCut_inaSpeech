# 本地验收记录

日期：2026-10-16。工作目录：用户指定的独立插件项目根目录。以下记录对应首次本地开发与验收；该阶段未初始化 Git、创建提交、推送、创建 PR 或修改主程序工作区。后续仓库接入与首次推送作为独立操作执行，项目自有代码采用 MIT 许可。

## 功能与测试

- Linux 完整测试：`pytest tests/ --host-source <宿主快照> --capture=sys --cov=blc_inaspeech --cov-fail-under=75`，**74 项通过，覆盖率 90.88%，无跳过**。
- Windows 专项：**73 项通过，无跳过**。包含所有客户端、解码、HTTP、模型准备、配置/文档、基准辅助逻辑和独立插件包导入测试；默认事件循环宿主生命周期测试单独在 Linux 完成。
- 官方真实 CNN：通过 `scripts/verify_runtime.py` 验证 CPU 分类、固定模型、HTTP 就绪/鉴权、完整区间覆盖和缓存一致性。
- Windows 实际调用：通过 `scripts/verify_host_service.py`，使用真实宿主 DTO、本机 FFmpeg、正式 Compose 中的真实模型；原转写和规则分保留。
- 容器：实际启动为 healthy；UID 1000；源码、依赖及模型目录创建文件被拒绝；运行数据目录可写。通过 `scripts/verify_container.py` 验证。
- 停止：SIGTERM 正常退出码 0，无 OOM，未留下推理临时音频目录，验收服务已停止。
- 性能：3 次串行测量，全部分类区间一致；详见 [性能报告](PERFORMANCE.md) 及原始 JSON。

## 静态、依赖和构建命令

```text
ruff check src tests scripts plugin
ruff format --check src tests scripts plugin
mypy --no-incremental
mypy --no-incremental --platform linux
uv lock --check --python <开发环境Python>
uv build --python <开发环境Python>
python scripts/build_plugin.py
docker compose config --quiet
docker compose -f compose.yaml -f compose.setup.yaml config --quiet
docker compose -f compose.yaml -f compose.gpu.yaml config --quiet
```

构建制品包含完整插件 ZIP、wheel 和含脚本/插件入口/说明文档的源码包。最终文件版本、大小、SHA-256、CRC32 在 `dist/artifact-manifest.json`，只对实际最终字节计算。

开发环境 66 个第三方发行版与 Linux 实际安装的 69 个发行版，已知漏洞查询均为 0；Linux 依赖一致性检查通过。具体范围、查询方式和边界见 [安全说明](SECURITY.md)。自有可交付源码另做敏感信息扫描，不将本地真实令牌、模型和依赖缓存纳入发布范围。

## 环境问题及未验证范围

Windows 某次扩大测试在 Python 默认事件循环初始化的标准库 `socketpair.accept` 处超时，尚未进入插件业务调用。早期较小集的真实宿主生命周期曾通过；本轮没有据此宣称 Windows 全量通过。Linux 完整测试及 Windows 专项、实际 Windows 到真实模型服务的调用分别提供证据，未通过删测试或改宿主事件循环掩盖环境问题。

Docker 内把临时目录放在 Windows 挂载盘时，pytest 默认 fd 捕获遇到 `TemporaryFile.truncate` 的文件系统行为差异，因此使用 `--capture=sys` 保留全部断言。容器在线 pip-audit 遇到 PyPI DNS 错误，改为导出真实安装列表后在 Windows 查询相同精确版本，没有跳过 Linux 包。

本机没有 NVIDIA GPU，CUDA 正向运行及性能未实测。缺失 GPU 的拒绝静默降级分支有回归测试。直接 WSL Ubuntu、macOS、ARM64 和多路中文直播准确率均没有本轮通过结论。

## 证据位置与保密

原始日志、宿主验收快照、真实模型、依赖环境、源文件审查清单与检查点都在本项目 `.local/`。它们包含本地服务令牌及运行数据，已被 `.gitignore` 排除，不应上传 Git。可公开的样本哈希、测量结果及响应例子在 `docs/evidence/`。

测试中读过的主程序源码没有合入此项目的交付制品。独立源码包不含宿主 `.env`、数据库、媒体或缓存。Docker 自身的镜像及引擎内部数据由 Docker 管理，项目新文件均保留在指定目录。
