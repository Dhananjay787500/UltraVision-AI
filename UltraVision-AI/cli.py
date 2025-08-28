#!/usr/bin/env python3
"""
Command-line interface for the AI Video Upscaler.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from video_upscaler.config import config
from video_upscaler.pipeline.upscaler import VideoUpscaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_video(
    input_path: Path,
    output_path: Optional[Path],
    model_name: str,
    resolution: str,
    keep_audio: bool,
    tile_size: int,
    use_gpu: bool,
    fp16: bool
) -> bool:
    """Process a single video file."""
    try:
        # Initialize the upscaler
        upscaler = VideoUpscaler(
            model_name=model_name,
            output_resolution=resolution,
            keep_audio=keep_audio,
            device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu',
            tile_size=tile_size,
            fp16=fp16
        )
        
        # Process the video
        logger.info(f"Processing: {input_path}")
        result_path = upscaler.process_video(
            input_path=input_path,
            output_path=output_path
        )
        
        if result_path and result_path.exists():
            logger.info(f"Successfully processed video. Output saved to: {result_path}")
            return True
        else:
            logger.error(f"Failed to process video: {input_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}", exc_info=True)
        return False

def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    resolution: str,
    keep_audio: bool,
    tile_size: int,
    use_gpu: bool,
    fp16: bool,
    recursive: bool = False
) -> None:
    """Process all supported videos in a directory."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    patterns = [f'*{ext}' for ext in config.SUPPORTED_INPUT_FORMATS]
    
    if recursive:
        for pattern in patterns:
            video_files.extend(input_dir.rglob(pattern))
    else:
        for pattern in patterns:
            video_files.extend(input_dir.glob(pattern))
    
    if not video_files:
        logger.warning(f"No supported video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    success_count = 0
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Determine output path
        rel_path = video_path.relative_to(input_dir) if recursive else video_path.name
        output_path = output_dir / rel_path.with_stem(f"{video_path.stem}_upscaled")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the video
        if process_single_video(
            input_path=video_path,
            output_path=output_path,
            model_name=model_name,
            resolution=resolution,
            keep_audio=keep_audio,
            tile_size=tile_size,
            use_gpu=use_gpu,
            fp16=fp16
        ):
            success_count += 1
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(video_files)} videos.")

def main():
    """Parse command-line arguments and run the appropriate function."""
    # Import torch here to avoid slow imports when just showing help
    import torch
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='AI Video Upscaler - Enhance and upscale videos using AI')
    
    # Input/output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        type=Path,
        help='Path to input video file',
        metavar='FILE'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Directory containing input videos',
        metavar='DIR'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Path to output video file (default: auto-generated)',
        metavar='FILE'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=config.OUTPUT_DIR,
        help=f'Directory to save output videos (default: {config.OUTPUT_DIR})',
        metavar='DIR'
    )
    
    # Processing options
    parser.add_argument(
        '-r', '--resolution',
        choices=['1080p', '1440p', '4k', '8k'],
        default='4k',
        help='Target resolution (default: 4k)'
    )
    parser.add_argument(
        '-m', '--model',
        choices=['realesrgan-x4plus', 'realesrgan-x4plus-anime'],
        default='realesrgan-x4plus',
        help='AI model to use (default: realesrgan-x4plus)'
    )
    parser.add_argument(
        '--no-audio',
        action='store_false',
        dest='keep_audio',
        help='Remove audio from output video'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=0,
        help='Tile size for processing (0 for no tiling, reduce if out of memory) (default: 0)',
        metavar='SIZE'
    )
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--cpu',
        action='store_false',
        dest='use_gpu',
        help='Use CPU even if GPU is available'
    )
    perf_group.add_argument(
        '--no-fp16',
        action='store_false',
        dest='fp16',
        help='Disable FP16 precision (slower but more accurate)'
    )
    
    # Directory processing options
    dir_group = parser.add_argument_group('Directory Processing Options')
    dir_group.add_argument(
        '--recursive',
        action='store_true',
        help='Process videos in subdirectories recursively'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file/directory exists
    if args.input and not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    if args.input_dir and not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Check if output directory exists and is writable
    if args.output_dir:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = args.output_dir / '.write_test'
            test_file.touch()
            test_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Cannot write to output directory {args.output_dir}: {e}")
            sys.exit(1)
    
    # Process single file or directory
    try:
        if args.input:
            # Process single file
            process_single_video(
                input_path=args.input,
                output_path=args.output,
                model_name=args.model,
                resolution=args.resolution,
                keep_audio=args.keep_audio,
                tile_size=args.tile_size,
                use_gpu=args.use_gpu and torch.cuda.is_available(),
                fp16=args.fp16 and torch.cuda.is_available()
            )
        else:
            # Process directory
            process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_name=args.model,
                resolution=args.resolution,
                keep_audio=args.keep_audio,
                tile_size=args.tile_size,
                use_gpu=args.use_gpu and torch.cuda.is_available(),
                fp16=args.fp16 and torch.cuda.is_available(),
                recursive=args.recursive
            )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
