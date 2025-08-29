# 🎥 AI Video Upscaler

A powerful, user-friendly application for enhancing and upscaling videos using AI. This tool leverages state-of-the-art deep learning models to upscale videos to higher resolutions (up to 8K) while improving quality and reducing noise.


## ✨ Features

- Upscale videos to 1080p, 2K (1440p), 4K (2160p), and 8K (4320p)
- Multiple AI models for different content types (General Purpose and Anime/Cartoon)
- Preserves original audio quality
- GPU acceleration support for faster processing
- User-friendly Streamlit web interface
- Batch processing support
- Real-time progress tracking

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- NVIDIA GPU with CUDA support (recommended for faster processing)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Dhananjay787500/UltraVision-AI.git
   cd UltraVision-AI
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   - **Windows**: Download from [FFmpeg's official website](https://ffmpeg.org/download.html) and add to PATH
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

## 🖥️ Usage

### Web Interface (Recommended)

Run the Streamlit web interface:
```bash
streamlit run UltraVision_AI/ui/streamlit_app.py
```

Then open your web browser to `http://localhost:8501`

### Command Line Interface

Process a single video:
```bash
python -m UltraVision_AI.cli --input input.mp4 --resolution 4k --model realesrgan-x4plus --output output.mp4
```

Process all videos in a directory:
```bash
python -m UltraVision_AI.cli --input-dir ./input_videos --resolution 4k --model realesrgan-x4plus --output-dir ./output_videos
```

### CLI Options

```
  -h, --help            show help message and exit
  --input INPUT         Path to input video file
  --input-dir INPUT_DIR
                        Directory containing input videos
  --output OUTPUT       Path to output video file (default: auto-generated)
  --output-dir OUTPUT_DIR
                        Directory to save output videos (default: ./output)
  --resolution {1080p,1440p,4k,8k}
                        Target resolution (default: 4k)
  --model {realesrgan-x4plus,realesrgan-x4plus-anime}
                        AI model to use (default: realesrgan-x4plus)
  --keep-audio          Keep original audio (default: True)
  --tile-size TILE_SIZE
                        Tile size for processing (0 for no tiling) (default: 0)
  --gpu                 Use GPU acceleration if available (default: True)
  --fp16                Use FP16 precision (faster, requires CUDA) (default: True)
```

## 🧠 AI Models

This project includes the following pre-trained models:

1. **RealESRGAN x4+ (General Purpose)** - Best for real-world videos and photos
2. **RealESRGAN x4+ (Anime/Cartoon)** - Optimized for anime and cartoon content

Models will be automatically downloaded on first use and cached for future use.

## 🏗️ Project Structure

```
UltraVision_AI/
├── config.py             # Application configuration
├── models/               # AI model implementations
│   ├── __init__.py
│   ├── base.py           # Base model interface
│   └── realesrgan.py     # RealESRGAN implementation
├── pipeline/             # Video processing pipeline
│   ├── __init__.py
│   ├── io.py            # Video I/O operations
│   └── upscaler.py      # Main upscaling logic
└── ui/
    └── streamlit_app.py # Web interface
```

## 🛠️ Customization

### Environment Variables

Create a `.env` file in the project root to customize settings:

```ini
# GPU settings
USE_GPU=true
FP16=true

# Processing settings
TILE_SIZE=512
TILE_PAD=10
BATCH_SIZE=4

# Paths
MODELS_DIR=./models
OUTPUT_DIR=./output
TEMP_DIR=./temp
LOGS_DIR=./logs
```

### Adding New Models

1. Create a new model class in `models/` that inherits from `BaseModel`
2. Register the model in `models/__init__.py`
3. Update the configuration and UI to support the new model

## 📊 Performance

Performance depends on your hardware configuration. Here are some benchmarks:

| Hardware | Resolution | FPS (GPU) | FPS (CPU) |
|----------|------------|-----------|-----------|
| RTX 3090 | 1080p → 4K | 12-15 FPS | 0.5-1 FPS |
| RTX 2080 Ti | 1080p → 4K | 8-10 FPS | 0.3-0.8 FPS |
| GTX 1080 Ti | 1080p → 4K | 4-6 FPS | 0.2-0.5 FPS |
| CPU Only | 1080p → 4K | N/A | 0.05-0.1 FPS |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for the upscaling models
- [Streamlit](https://streamlit.io/) for the web interface
- [FFmpeg](https://ffmpeg.org/) for video processing
- [PyTorch](https://pytorch.org/) for deep learning

---

<div align="center">
  Made with ❤️ by Dhananjay Kamble | <a href="https://github.com/Dhananjay787500/UltraVision-AI">GitHub</a>
</div>
