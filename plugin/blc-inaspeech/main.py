"""宿主只导入轻量适配器，不会导入 TensorFlow。"""

from blc_inaspeech.adapter import Plugin

__all__ = ["Plugin"]
