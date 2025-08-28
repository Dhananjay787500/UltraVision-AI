"""AI models for video upscaling and enhancement."""

from .base import BaseModel
from .realesrgan import RealESRGANModel

__all__ = ['BaseModel', 'RealESRGANModel']
