# 配置与结果读取

## 宿主插件设置

设置通过宿主 `PluginSetting` 声明、校验并持久化，修改后下一次调用读取新值，无需服务端动态安装依赖。

| 键 | 默认值 | 校验与含义 |
| --- | --- | --- |
| `endpoint` | `http://127.0.0.1:8766` | 无路径、凭据、查询或片段的 origin；禁止控制字符、零端口及重定向 |
| `token` | 空 | 必须由管理员填写 64 位小写十六进制服务令牌；不使用默认秘密 |
| `media_root` | 空 | 必须显式指定宿主可访问的媒体目录；目录外文件和符号链接媒体被拒绝 |
| `ffmpeg` | `ffmpeg` | 程序名或完整路径，通过参数数组执行，不使用 shell |
| `timeout_s` | `120` | HTTP 请求等待秒数，范围 1–300 |
| `decode_timeout_s` | `30` | 解码等待秒数，范围 1–120 |
| `allow_network_http` | `false` | 仅为管理员认可的 Docker 专用网络显式允许非环回 HTTP；远端建议 HTTPS |

明文环回地址必须使用 IP，例如 `127.0.0.1` 或 `::1`；不把可能受 DNS/hosts 影响的 `localhost` 视为已经验证的环回 IP。当前服务默认监听 IPv4，客户端允许连接已有的 IPv6 服务。

令牌与主程序管理员密码独立。宿主现有的 password 字段只隐藏回显，设置仍在宿主本地数据库；空白保存不覆盖旧值。保护数据库、备份及 Windows 目录 ACL，不要将令牌贴到 issue、日志或截图。

## 独立服务参数

```text
blc-inaspeech serve --host 127.0.0.1 --port 8766 --model-dir .local/models --token-file .local/inaspeech.token --device cpu --batch-size 128 --threads 2 --cache-entries 32
```

- `--device`：`cpu` 或 `cuda`；CUDA 缺失时报错。
- `--batch-size`：1–1024，默认 128。增大 batch 可能提高吞吐，也会提高内存或显存占用；不是所有硬件都更快。
- `--threads`：1–32，默认 2。启动前限制 TensorFlow、OpenMP、OpenBLAS、MKL 和 NumExpr 的计算线程配置；TensorFlow 仍可能创建其他辅助线程，不代表进程总线程数等于 2。
- `--cache-entries`：0–128，默认 32；0 禁用缓存。只缓存分类结果，按完整 WAV 内容哈希，无磁盘音频缓存，重启失效。
- `--model-dir` 和 `--token-file`：由管理员显式指定的服务端文件，不来自 HTTP 请求。

同时只有一个模型调用，监听队列有上限。默认 Docker 配额为 2 CPU、4 GiB 内存、128 个进程/线程额度。调整使用 `.env.example` 中的变量；它们只由 Compose 读取，不是宿主 `.env` 的配置项。

直接 Linux/WSL 运行依赖 CLI 参数，`.env` 不会自动改变这些参数。受信任端点的 HTTP 等待和服务计算时长不是同一概念：客户端超时会回退，服务端可能仍在完成当前有界音频的计算。

## 分析语义

本插件在当前公开高光接口中运行，时点为 ASR 之后。没有转写前回调，不会跳过或加速已经执行的 ASR，不负责创建热点事件或候选。

房间模式 `off` 返回 off、不解码、不调用服务。其他模式均只返回 shadow 分类；即使房间请求 Champion，也不替换规则分，不虚构高光概率。审核反馈方法是明确的无副作用操作，不收集训练样本。

分类仅表示片段内声音类型：

| 标签 | 含义 |
| --- | --- |
| `speech` | 上游的人声分类，背景有音乐时仍可能归为人声 |
| `music` | 上游的音乐分类，唱歌也可能归入音乐 |
| `noise` | 上游的噪声分类 |
| `silence` | 上游 `noEnergy`，表示能量活动判断，不是精确静音真值 |

结果不会自动过滤音乐片段。音乐/唱歌、人声叠加、中文口音、游戏音效等情况需要业务样本评估，不能根据法语或公开混合音频测试推断准确率。

## 存储与时间

宿主保存的路径为：

```text
highlight_plugin.prediction.metadata.audio_classification
highlight_plugin.prediction.feature_values.inaspeech_speech_ratio
highlight_plugin.prediction.feature_values.inaspeech_music_ratio
highlight_plugin.prediction.feature_values.inaspeech_noise_ratio
highlight_plugin.prediction.feature_values.inaspeech_silence_ratio
```

宿主如何保留被初筛拒绝片段的观测由宿主流程决定，不能认为每次调用都有可见候选页面。现有 UI 没有本插件专属的分类时间线；CLI 可直接读取 JSON，宿主候选特征和插件观测中保留结构化结果。

时间原点是送入模型的片段媒体起点，单位秒，区间为半开 `[start_s, end_s)`。模型输出采用约 20 ms 栅格，末尾仅对齐最多 40 ms 的栅格偏差，并裁到真实音频长度；更大的缺口拒绝。不会盲目把媒体时间与场次墙钟时间相加。

类别占比按已验证的区间时长线性汇总，总和为 1。分类概率和高光概率均未从该占比推导。

`inference_s` 是该次未缓存推理的服务内部耗时；命中缓存时保留原结果的这个值，`cache_hit=true` 表示没有重新推理。不能将它当成缓存 HTTP 调用的实际耗时。测量端到端延迟需另计解码、传输、排队及宿主开销。
