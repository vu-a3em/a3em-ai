"""Audio file conversion and processing utilities."""

import os
import subprocess
from typing import Optional


def convert_to_wav(src_path: str, 
                  target_sr: int,
                  dest_path: Optional[str] = None,
                  remove_original: bool = True) -> str:
    """Convert an audio file to WAV format at the requested sample rate using ffmpeg.
    
    Args:
        src_path: Path to source audio file
        target_sr: Target sample rate in Hz
        dest_path: Optional destination path. If None, replaces extension with .wav
        remove_original: Whether to remove the original file after conversion (only if not already .wav)
    
    Returns:
        Path to the created WAV file
    
    Raises:
        RuntimeError: If ffmpeg fails or is not found
    """
    if dest_path is None:
        dest_path = os.path.splitext(src_path)[0] + ".wav"
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-i",
        src_path,
        "-ar",
        str(int(target_sr)),
        dest_path,
    ]
    
    try:
        completed = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        
        # Check if the output file was created
        if not os.path.exists(dest_path):
            raise RuntimeError(f"ffmpeg conversion failed: output file {dest_path} not created")
        
        # Remove the original file if requested and it's not already a WAV
        if remove_original and os.path.splitext(src_path)[1].lower() != ".wav":
            os.remove(src_path)
            
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on system; required for audio conversion")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='ignore') if e.stderr is not None else str(e)
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
    
    return dest_path
