from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from pathlib import Path

class BaseModel(ABC):
    """Base class for all upscaling models."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """Initialize the model.
        
        Args:
            model_path: Path to the model weights. If None, will download automatically.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model = None
        self.scale = 1
        self.name = "base_model"
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights."""
        pass
    
    @abstractmethod
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Upscale a single image.
        
        Args:
            img: Input image as a numpy array in BGR format.
            
        Returns:
            Upscaled image as a numpy array in BGR format.
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_scale_factor(self) -> int:
        """Get the scale factor of the model (e.g., 2 for 2x upscaling)."""
        return self.scale
    
    def __del__(self):
        """Cleanup when the object is deleted."""
        self.unload_model()
