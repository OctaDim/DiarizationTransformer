import subprocess
import sys

import pyannote.core
import torch
import torchaudio
# import torchcodec  # Issue here!!!

def get_features_versions():
    print(f"Python version: {sys.version}")
    print(f"Pyannote version: {pyannote.core.__version__}")
    # print(f"TorchCodec версия: {torchcodec.__version__}")  # Issue here!!!
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchAudio version: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    cuda_is_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    print(f"CUDA available: {cuda_is_available}, version: {cuda_version}")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device name: {torch.device(device_name)}")

    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_version = result.stdout.split('\n')[0].split(' ')[2]
            print(f"System FFmpeg version: {ffmpeg_version}")
        else:
            print("System FFmpeg: Not available")
    except FileNotFoundError:
        print("System FFmpeg: Not installed")
