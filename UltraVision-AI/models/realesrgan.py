import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from .base import BaseModel

class RealESRGANModel(BaseModel):
    """RealESRGAN model for image and video upscaling."""
    
    MODEL_URLS = {
        'realesrgan-x4plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'model': RRDBNet,
            'scale': 4,
            'params': {
                'num_in_ch': 3,
                'num_out_ch': 3,
                'num_feat': 64,
                'num_block': 23,
                'num_grow_ch': 32,
                'scale': 4
            }
        },
        'realesrgan-x4plus-anime': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'model': RRDBNet,
            'scale': 4,
            'params': {
                'num_in_ch': 3,
                'num_out_ch': 3,
                'num_feat': 64,
                'num_block': 6,
                'num_grow_ch': 32,
                'scale': 4
            }
        },
    }
    
    def __init__(
        self,
        model_name: str = 'realesrgan-x4plus',
        model_path: Optional[str] = None,
        device: str = 'cuda',
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        fp16: bool = True
    ):
        """Initialize RealESRGAN model.
        
        Args:
            model_name: Name of the model to use (from MODEL_URLS).
            model_path: Path to the model weights. If None, will download automatically.
            device: Device to run the model on ('cuda' or 'cpu').
            tile: Tile size, 0 for no tile during testing.
            tile_pad: Tile padding.
            pre_pad: Pre padding size at each border.
            fp16: Use fp16 precision.
        """
        super().__init__(model_path, device)
        self.model_name = model_name
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp16 = fp16
        self.upsampler = None
        
        # Set model config based on model_name
        if model_name not in self.MODEL_URLS:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(self.MODEL_URLS.keys())}")
        
        model_config = self.MODEL_URLS[model_name]
        self.scale = model_config['scale']
        self.model_arch = model_config['model']
        self.model_params = model_config['params']
        self.model_url = model_config['url']
        self.name = model_name
        
        # Set model path
        if model_path is None:
            self.model_path = self._get_default_model_path()
    
    def _get_default_model_path(self) -> Path:
        """Get the default path to save/load the model."""
        from ...config import config
        model_dir = config.MODELS_DIR / 'realesrgan'
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{self.model_name}.pth"
    
    def load_model(self) -> None:
        """Load the RealESRGAN model."""
        if self.upsampler is not None:
            return
            
        # Download model if it doesn't exist
        if not self.model_path.exists():
            self._download_model()
        
        # Initialize model
        model = self.model_arch(**self.model_params)
        
        # Initialize the upsampler
        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.fp16,
            device=self.device
        )
    
    def _download_model(self) -> None:
        """Download the model weights if they don't exist."""
        import urllib.request
        import ssl
        
        print(f"Downloading {self.model_name} model from {self.model_url}...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SSL context to handle certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Download the model
        urllib.request.urlretrieve(self.model_url, self.model_path)
        print(f"Model downloaded to {self.model_path}")
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Upscale a single image.
        
        Args:
            img: Input image as a numpy array in BGR format.
            
        Returns:
            Upscaled image as a numpy array in BGR format.
        """
        if self.upsampler is None:
            self.load_model()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        output, _ = self.upsampler.enhance(
            img_rgb,
            outscale=self.scale
        )
        
        # Convert back to BGR
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.upsampler is not None:
            if hasattr(self.upsampler, 'model'):
                del self.upsampler.model
            if hasattr(self.upsampler, 'device'):
                self.upsampler.device = None
            self.upsampler = None
        
        super().unload_model()
