from setuptools import setup, find_packages
import os

def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="UltraVision-AI",
    version="0.1.0",
    packages=find_packages(include=['UltraVision_AI', 'UltraVision_AI.*']),
    install_requires=read_requirements(),
    python_requires='>=3.8',
    
    # Metadata
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered video upscaling tool using Real-ESRGAN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/UltraVision-AI",
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Conversion",
    ],
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'ultra-vision=UltraVision_AI.cli:main',
        ],
    },
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        'UltraVision_AI': ['*.yaml', '*.json'],
    },
    
    # Dependencies for development
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.7b0',
            'isort>=5.0',
            'mypy>=0.910',
            'flake8>=3.9',
            'pre-commit>=2.13',
        ],
    },
)
