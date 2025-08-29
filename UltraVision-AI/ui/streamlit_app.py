import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_upscaler.config import config
from video_upscaler.models import RealESRGANModel
from video_upscaler.pipeline.upscaler import VideoUpscaler
from video_upscaler.pipeline.io import get_video_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="UltraVision-AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    .css-1v3fvcr {
        padding: 2rem 1rem 1rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
SUPPORTED_FORMATS = ['.mp4', '.mov', '.mkv', '.avi']
RESOLUTIONS = {
    '1080p': '1920x1080 (Full HD)',
    '1440p': '2560x1440 (2K/QHD)',
    '4k': '3840x2160 (4K/UHD)',
    '8k': '7680x4320 (8K/UHD2)'
}
MODELS = {
    'realesrgan-x4plus': 'RealESRGAN x4+ (General Purpose)',
    'realesrgan-x4plus-anime': 'RealESRGAN x4+ (Anime/Cartoon)'
}

def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def process_video(
    input_path: str,
    output_path: str,
    model_name: str,
    resolution: str,
    progress_bar,
    status_text
) -> Dict[str, Any]:
    """Process a video file with the specified settings."""
    try:
        # Initialize the upscaler
        upscaler = VideoUpscaler(
            model_name=model_name,
            output_resolution=resolution,
            keep_audio=True,
            device='cuda' if config.USE_GPU and torch.cuda.is_available() else 'cpu',
            tile_size=config.TILE_SIZE,
            tile_pad=config.TILE_PAD,
            fp16=config.FP16
        )
        
        # Process the video
        result_path = upscaler.process_video(
            input_path=input_path,
            output_path=output_path
        )
        
        if result_path and os.path.exists(result_path):
            return {
                'success': True,
                'output_path': str(result_path),
                'message': 'Video processing completed successfully!'
            }
        else:
            return {
                'success': False,
                'message': 'Video processing failed. Please check the logs for details.'
            }
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }

def main():
    """Main Streamlit application."""
    st.title("🎥 UltraVision-AI")
    st.caption("Enhance and upscale your videos using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "AI Model",
            options=list(MODELS.keys()),
            format_func=lambda x: MODELS[x],
            help="Select the AI model for upscaling"
        )
        
        # Resolution selection
        selected_resolution = st.selectbox(
            "Output Resolution",
            options=list(RESOLUTIONS.keys()),
            format_func=lambda x: f"{x.upper()} - {RESOLUTIONS[x]}",
            help="Select the target resolution for the output video"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.checkbox("Use GPU Acceleration", value=config.USE_GPU, disabled=not torch.cuda.is_available(), 
                       help="Enable GPU acceleration if available")
            st.checkbox("Keep Original Audio", value=True, key="keep_audio",
                      help="Keep the original audio in the output video")
            st.slider("Tile Size", min_value=0, max_value=1024, value=config.TILE_SIZE, step=32,
                    help="Tile size for processing. 0 for no tiling. Reduce if you run out of memory.")
            st.checkbox("Use FP16 Precision", value=config.FP16, 
                      help="Use half-precision floating point for faster processing (recommended for NVIDIA GPUs with Tensor Cores)")
    
    # Main content
    st.header("Upload Video")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=False,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_dir = config.TEMP_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = temp_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video info
        try:
            video_info = get_video_info(input_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            with col2:
                st.metric("Frame Rate", f"{video_info['frame_rate']:.2f} FPS")
            with col3:
                st.metric("Duration", format_duration(video_info['duration']))
            
            # Display video preview
            st.subheader("Preview")
            video_bytes = open(input_path, 'rb').read()
            st.video(video_bytes)
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                if not input_path.exists():
                    st.error("Error: Uploaded file not found. Please try again.")
                else:
                    # Create output directory
                    output_dir = config.OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{input_path.stem}_upscaled{input_path.suffix}"
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process the video
                    status_text.text("Starting video processing...")
                    result = process_video(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        model_name=selected_model,
                        resolution=selected_resolution,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    
                    if result['success']:
                        progress_bar.progress(100)
                        status_text.success(result['message'])
                        
                        # Show download button
                        with open(result['output_path'], 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Processed Video",
                                data=f,
                                file_name=os.path.basename(result['output_path']),
                                mime="video/mp4"
                            )
                        
                        # Show before/after comparison
                        st.subheader("Before & After")
                        
                        # Extract first frame from original video
                        cap_original = cv2.VideoCapture(str(input_path))
                        ret_original, frame_original = cap_original.read()
                        cap_original.release()
                        
                        # Extract first frame from processed video
                        cap_processed = cv2.VideoCapture(result['output_path'])
                        ret_processed, frame_processed = cap_processed.read()
                        cap_processed.release()
                        
                        if ret_original and ret_processed:
                            # Resize for display
                            max_height = 400
                            h, w = frame_original.shape[:2]
                            scale = max_height / h
                            display_size = (int(w * scale), int(h * scale))
                            
                            frame_original = cv2.resize(frame_original, display_size)
                            frame_processed = cv2.resize(frame_processed, display_size)
                            
                            # Convert to RGB for display
                            frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
                            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                            
                            # Display side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(frame_original, caption="Original", use_column_width=True)
                            with col2:
                                st.image(frame_processed, caption="Processed", use_column_width=True)
                        
                    else:
                        status_text.error(result['message'])
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Error in main: {str(e)}", exc_info=True)
    
    # Add some spacing
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<p>🎥 UltraVision-AI v0.1.0 | "
        "<a href='https://github.com/Dhananjay787500/UltraVision-AI' target='_blank'>GitHub</a> | "
        "<a href='https://github.com/Dhananjay787500/UltraVision-AI/issues' target='_blank'>Report Issues</a></p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Import torch here to avoid streamlit caching issues
    import torch
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU. Processing will be slow.")
    
    # Run the app
    main()


