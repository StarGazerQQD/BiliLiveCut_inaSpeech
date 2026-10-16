# 第三方来源与许可

本项目自有代码采用根目录 [MIT 许可](../LICENSE)。依赖与模型各自受上游许可约束，项目自身的 MIT 不覆盖全部第三方制品。

| 组件 | 使用方式 | 从实际发行元数据核对的许可 |
| --- | --- | --- |
| inaSpeechSegmenter 0.8.0 | 独立环境安装，调用官方 `smn` | MIT |
| TensorFlow 2.21.0 | 独立环境推理 | Apache 2.0 |
| NumPy 2.5.4 | 上游数值计算 | BSD-3-Clause，发行包还声明 0BSD/MIT/Zlib/CC0-1.0 组成部分 |
| scikit-image 0.26.0 | 上游特征窗口处理 | 主要为 BSD-3-Clause，具体文件另有声明 |
| SoundFile 0.14.0 | 上游读取规范 WAV | Python 部分 BSD-3-Clause；其原生依赖另按发行包许可 |
| ONNX Runtime GPU 1.31.0 | 官方 inaSpeech 发行声明的依赖；本版 CNN 未使用其推理路径 | MIT |
| Pyro4 4.82 | 官方发行声明依赖；本项目不启动 Pyro 服务 | MIT |
| NVIDIA CUDA/cuDNN wheel | 官方 TensorFlow CUDA extra 的传递依赖，CPU 安装也保留 | NVIDIA 专有许可，不能笼统归为 MIT |
| uv 0.12.18 官方镜像 | 部署工具和 Python 运行环境 | uv 项目及镜像内组件分别遵循各自许可 |

详细已安装版本及许可证元数据保存在本地 `runtime-licenses.json` 验收报告中。此文件位于忽略的验收目录，不是完整法律意见或所有二进制组成部分的逐文件审计。

官方模型从 [上游 models Release](https://github.com/ina-foss/inaSpeechSegmenter/releases/tag/models) 获取，本项目只分发 URL、大小和固定内容哈希，不再分发权重。模型使用应遵循上游授权和适用限制；不额外宣称权重具有单独的发布签名或训练数据保证。

性能测试使用上游公开 MUSAN 混合音频，来源说明见 [上游 media README](https://github.com/ina-foss/inaSpeechSegmenter/blob/7a41b809c3cc6d8d36b24e70784779f8d16f7b5e/media/README.txt)。本项目仅保留样本哈希和测量结果，没有将测试音频加入插件或源码制品。
