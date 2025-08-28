import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Application settings
    APP_NAME = "AI Video Upscaler"
    VERSION = "0.1.0"
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Video processing settings
    SUPPORTED_INPUT_FORMATS = ['.mp4', '.mov', '.mkv', '.avi']
    DEFAULT_OUTPUT_FORMAT = '.mp4'
    DEFAULT_RESOLUTION = '1080p'
    
    # Model settings
    DEFAULT_MODEL = 'realesrgan-x4plus'
    AVAILABLE_MODELS = {
        'realesrgan-x4plus': 'RealESRGAN x4+',
        'realesrgan-x4plus-anime': 'RealESRGAN x4+ (Anime)',
    }
    
    # GPU/CPU settings
    USE_GPU: bool = os.getenv('USE_GPU', 'true').lower() == 'true'
    FP16: bool = os.getenv('FP16', 'true').lower() == 'true'
    
    # Processing settings
    TILE_SIZE: int = int(os.getenv('TILE_SIZE', '512'))
    TILE_PAD: int = int(os.getenv('TILE_PAD', '10'))
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '4'))
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        # Create required directories
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get the appropriate device (cuda or cpu) based on availability."""
        import torch
        return 'cuda' if self.USE_GPU and torch.cuda.is_available() else 'cpu'
    
    def get_resolution_dimensions(self, resolution: str) -> tuple[int, int]:
        """Get width and height for a given resolution string."""
        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '1440p': (2560, 1440),  # 2K
            '4k': (3840, 2160),     # 4K UHD
            '8k': (7680, 4320),     # 8K UHD
        }
        return resolution_map.get(resolution.lower(), (1920, 1080))

# Create a global config instance
config = Config()
