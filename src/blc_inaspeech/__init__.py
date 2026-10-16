"""独立 inaSpeech 分类服务及 BiliLiveCut 公共插件适配器。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("blc-inaspeech")
except PackageNotFoundError:
    __version__ = "0.1.0"
