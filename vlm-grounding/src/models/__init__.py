from .base import Capability, ImageContext, UnsupportedCapability, VLM, VLMOutput  # noqa: F401
from .registry import available_models, build_model, register_model  # noqa: F401
from . import dummy  # noqa: F401  (registers 'dummy_brightness')
from . import hf_blip2  # noqa: F401  (registers 'hf_blip2'; torch/transformers imported lazily)
