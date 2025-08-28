import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from tqdm import tqdm
import shutil

from ...config import config
from ...models import RealESRGANModel
from .io import (
    get_video_info,
    extract_frames,
    extract_audio,
    create_video_from_frames,
    combine_video_audio
)

logger = logging.getLogger(__name__)

class VideoUpscaler:
    """Main class for video upscaling pipeline."""
    
    def __init__(
        self,
        model_name: str = 'realesrgan-x4plus',
        output_resolution: str = '4k',
        keep_audio: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        tile_size: int = 0,
        tile_pad: int = 10,
        fp16: bool = True,
    ):
        """Initialize the video upscaler.
        
        Args:
            model_name: Name of the model to use for upscaling.
            output_resolution: Target resolution (e.g., '1080p', '2k', '4k', '8k').
            keep_audio: Whether to keep the original audio in the output video.
            output_dir: Directory to save output files. Uses config.OUTPUT_DIR if None.
            temp_dir: Directory for temporary files. Uses config.TEMP_DIR if None.
            device: Device to run the model on ('cuda' or 'cpu').
            tile_size: Tile size for processing. 0 for no tiling.
            tile_pad: Tile padding.
            fp16: Use FP16 precision if available.
        """
        self.model_name = model_name
        self.output_resolution = output_resolution.lower()
        self.keep_audio = keep_audio
        self.device = device or config.device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = fp16 and (self.device == 'cuda')
        
        # Set up directories
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        self.temp_dir = Path(temp_dir) if temp_dir else config.TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._load_model()
        
        # Get target dimensions
        self.target_width, self.target_height = config.get_resolution_dimensions(self.output_resolution)
    
    def _load_model(self) -> RealESRGANModel:
        """Load the upscaling model."""
        logger.info(f"Loading {self.model_name} model...")
        model = RealESRGANModel(
            model_name=self.model_name,
            device=self.device,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            fp16=self.fp16
        )
        model.load_model()
        return model
    
    def _get_output_path(self, input_path: Union[str, Path]) -> Path:
        """Generate output path based on input path and settings."""
        input_path = Path(input_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}__{self.output_resolution}__{self.model_name}__{timestamp}{config.DEFAULT_OUTPUT_FORMAT}"
        return self.output_dir / output_filename
    
    def _cleanup_temp_files(self, *paths: Union[str, Path]) -> None:
        """Clean up temporary files and directories."""
        for path in paths:
            path = Path(path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with the upscaling model."""
        # Convert BGR to RGB if needed by the model
        if self.model.model_arch.__name__ == 'RRDBNet':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_rgb, _ = self.model.upsampler.enhance(frame_rgb, outscale=self.model.scale)
            output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        else:
            output = self.model.predict(frame)
        
        # Resize to target resolution if needed
        if output.shape[0] != self.target_height or output.shape[1] != self.target_width:
            output = cv2.resize(
                output, 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LANCZOS4
            )
        
        return output
    
    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """Process a video file.
        
        Args:
            input_path: Path to the input video file.
            output_path: Path to save the output video. If None, will be generated automatically.
            
        Returns:
            Path to the processed video file, or None if processing failed.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None
        
        # Set up output path
        if output_path is None:
            output_path = self._get_output_path(input_path)
        else:
            output_path = Path(output_path)
        
        logger.info(f"Processing video: {input_path.name}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # Create temporary directories
        temp_dir = self.temp_dir / f"temp_{input_path.stem}"
        frames_dir = temp_dir / "frames"
        processed_frames_dir = temp_dir / "processed_frames"
        audio_path = temp_dir / "audio.aac"
        
        try:
            # Get video info
            video_info = get_video_info(input_path)
            frame_rate = video_info['frame_rate']
            total_frames = video_info['total_frames']
            
            logger.info(f"Input video: {video_info['width']}x{video_info['height']} @ {frame_rate:.2f} fps, {total_frames} frames")
            logger.info(f"Target resolution: {self.target_width}x{self.target_height}")
            
            # Extract audio if needed
            if self.keep_audio:
                logger.info("Extracting audio...")
                if not extract_audio(input_path, audio_path):
                    logger.warning("Failed to extract audio. Output will not have audio.")
                    self.keep_audio = False
            
            # Extract frames
            logger.info("Extracting frames...")
            frame_paths = extract_frames(
                input_path,
                frames_dir,
                num_threads=4
            )
            
            if not frame_paths:
                logger.error("No frames were extracted from the video.")
                return None
            
            # Process frames
            logger.info("Upscaling frames...")
            processed_frames_dir.mkdir(parents=True, exist_ok=True)
            processed_frame_paths = []
            
            for i, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    continue
                
                # Process frame
                try:
                    processed_frame = self._process_frame(frame)
                    
                    # Save processed frame
                    output_path = processed_frames_dir / f"frame_{i:08d}.jpg"
                    cv2.imwrite(str(output_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    processed_frame_paths.append(output_path)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {str(e)}")
                    continue
            
            if not processed_frame_paths:
                logger.error("No frames were processed successfully.")
                return None
            
            # Create video from processed frames
            logger.info("Creating output video...")
            temp_video_path = temp_dir / "output_no_audio.mp4"
            
            if not create_video_from_frames(
                frame_paths=processed_frame_paths,
                output_path=temp_video_path,
                frame_rate=frame_rate,
                codec='libx264',
                crf=18,
                preset='medium',
                pixel_format='yuv420p'
            ):
                logger.error("Failed to create output video.")
                return None
            
            # Add audio if needed
            if self.keep_audio and audio_path.exists():
                logger.info("Adding audio to output video...")
                if not combine_video_audio(
                    video_path=temp_video_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    copy_audio=True
                ):
                    logger.warning("Failed to add audio to output video. Saving without audio.")
                    shutil.move(temp_video_path, output_path)
            else:
                shutil.move(temp_video_path, output_path)
            
            logger.info(f"Processing complete. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            return None
            
        finally:
            # Clean up temporary files
            if temp_dir.exists():
                self._cleanup_temp_files(temp_dir)
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            self.model.unload_model()
