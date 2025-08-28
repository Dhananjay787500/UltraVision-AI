import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...config import config

def get_video_info(video_path: Union[str, Path]) -> Dict:
    """Get video metadata using ffprobe.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Dictionary containing video metadata.
    """
    video_path = str(video_path)
    
    # Get video info using ffprobe
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames,codec_name',
        '-show_entries', 'format=filename,size,duration',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract stream info
        stream = info['streams'][0]
        format_info = info['format']
        
        # Calculate frame rate
        if 'r_frame_rate' in stream:
            num, den = map(float, stream['r_frame_rate'].split('/'))
            frame_rate = num / den if den != 0 else 0
        else:
            frame_rate = 0
        
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'frame_rate': frame_rate,
            'duration': float(stream.get('duration', 0)) or float(format_info.get('duration', 0)),
            'total_frames': int(stream.get('nb_frames', 0)) or int(float(format_info.get('duration', 0)) * frame_rate),
            'codec': stream.get('codec_name', ''),
            'size_bytes': int(format_info.get('size', 0)),
            'filename': os.path.basename(format_info.get('filename', video_path))
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError) as e:
        # Fallback to OpenCV if ffprobe fails
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / frame_rate if frame_rate > 0 else 0
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'frame_rate': frame_rate,
            'duration': duration,
            'total_frames': frame_count,
            'codec': 'unknown',
            'size_bytes': os.path.getsize(video_path),
            'filename': os.path.basename(video_path)
        }

def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    quality: int = 95,
    num_threads: int = 4
) -> List[str]:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        start_frame: Frame number to start extraction from.
        end_frame: Frame number to end extraction at (inclusive).
        quality: JPEG quality (1-100) for saving frames.
        num_threads: Number of threads to use for extraction.
        
    Returns:
        List of paths to extracted frame files.
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info['total_frames']
    
    # Set end_frame if not specified
    if end_frame is None or end_frame >= total_frames:
        end_frame = total_frames - 1
    
    # Calculate total frames to process
    total_frames_to_process = end_frame - start_frame + 1
    
    # Create frame paths
    frame_paths = [output_dir / f"frame_{i:08d}.jpg" for i in range(start_frame, end_frame + 1)]
    
    # Check if all frames already exist
    existing_frames = [p for p in frame_paths if p.exists()]
    if len(existing_frames) == total_frames_to_process:
        return [str(p) for p in frame_paths]
    
    # Function to process a single frame
    def process_frame(frame_num: int) -> str:
        output_path = output_dir / f"frame_{frame_num:08d}.jpg"
        if output_path.exists():
            return str(output_path)
            
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return str(output_path) if ret else ""
    
    # Process frames in parallel
    frame_paths = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, i) for i in range(start_frame, end_frame + 1)]
        
        # Use tqdm to show progress
        with tqdm(total=total_frames_to_process, desc="Extracting frames") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    frame_paths.append(result)
                pbar.update(1)
    
    # Sort frame paths by frame number
    frame_paths.sort()
    return frame_paths

def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_codec: str = 'aac',
    audio_bitrate: str = '192k',
    sample_rate: int = 44100,
    channels: int = 2
) -> bool:
    """Extract audio from a video file.
    
    Args:
        video_path: Path to the video file.
        output_path: Path to save the extracted audio.
        audio_codec: Audio codec to use (e.g., 'aac', 'mp3', 'pcm_s16le').
        audio_bitrate: Audio bitrate (e.g., '192k').
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        
    Returns:
        True if extraction was successful, False otherwise.
    """
    video_path = str(video_path)
    output_path = str(output_path)
    
    # Check if output file already exists
    if os.path.exists(output_path):
        return True
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-vn',  # Disable video
        '-c:a', audio_codec,
        '-b:a', audio_bitrate,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        return False

def create_video_from_frames(
    frame_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    frame_rate: float,
    codec: str = 'libx264',
    crf: int = 18,
    preset: str = 'medium',
    pixel_format: str = 'yuv420p'
) -> bool:
    """Create a video from a sequence of frames.
    
    Args:
        frame_paths: List of paths to frame images.
        output_path: Path to save the output video.
        frame_rate: Frame rate of the output video.
        codec: Video codec to use.
        crf: Constant Rate Factor (lower = better quality, 18-28 is a good range).
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow).
        pixel_format: Pixel format of the output video.
        
    Returns:
        True if video creation was successful, False otherwise.
    """
    if not frame_paths:
        return False
    
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a temporary file with the list of frames
    temp_list_file = os.path.join(os.path.dirname(output_path), 'frame_list.txt')
    with open(temp_list_file, 'w') as f:
        for frame_path in frame_paths:
            f.write(f"file '{os.path.abspath(str(frame_path))}'\n")
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'concat',
        '-safe', '0',
        '-r', str(frame_rate),
        '-i', temp_list_file,
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', pixel_format,
        '-vsync', 'vfr',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)

def combine_video_audio(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    copy_audio: bool = True,
    audio_codec: Optional[str] = None,
    audio_bitrate: Optional[str] = None
) -> bool:
    """Combine a video file with an audio file.
    
    Args:
        video_path: Path to the video file (without audio).
        audio_path: Path to the audio file.
        output_path: Path to save the output video with audio.
        copy_audio: If True, copy the audio stream without re-encoding.
        audio_codec: Audio codec to use if re-encoding.
        audio_bitrate: Audio bitrate if re-encoding.
        
    Returns:
        True if combination was successful, False otherwise.
    """
    video_path = str(video_path)
    audio_path = str(audio_path)
    output_path = str(output_path)
    
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return False
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video stream without re-encoding
    ]
    
    # Handle audio stream
    if copy_audio:
        cmd.extend(['-c:a', 'copy'])  # Copy audio stream without re-encoding
    else:
        if audio_codec:
            cmd.extend(['-c:a', audio_codec])
        if audio_bitrate:
            cmd.extend(['-b:a', audio_bitrate])
    
    # Add output path
    cmd.extend([
        '-shortest',  # Finish encoding when the shortest input stream ends
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        output_path
    ])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e.stderr.decode()}")
        return False
