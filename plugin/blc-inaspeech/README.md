# inaSpeech 音频分类插件

此目录是 BiliLiveCut API 1 的轻量客户端，不包含 TensorFlow、模型或推理运行环境。

使用 `scripts/build_plugin.py` 生成的完整 ZIP 解压到 `PLUGIN_DIR` 后启用。构建后的目录须同时有 `plugin.json`、`main.py`、`blc_inaspeech/`。源码仓库中的这个入口目录不能单独复制使用。

先部署独立服务，再在插件设置填写服务地址、服务令牌和授权媒体根目录。完整部署说明位于项目源码分发包的 `README.md` 和 `docs/INSTALLATION.md`。

插件仅在转写后的高光接口中旁路记录人声、音乐、噪声和低能量区间；不覆盖原评分，不减少已执行的 ASR，不把分类占比当作高光概率。同一时间不能与其他高光评分插件启用。

更新前先停用；更新后重新扫描并启用。卸载前先停用，再移出此目录；不会删除宿主历史记录。
