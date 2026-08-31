import os
import shutil
import sys
from pathlib import Path

# Ensure ffmpeg from imageio_ffmpeg / venv is discoverable in PATH
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = Path(ffmpeg_exe).parent
    shim = ffmpeg_dir / "ffmpeg.exe"
    if not shutil.which("ffmpeg") and not shim.exists():
        shutil.copy2(ffmpeg_exe, shim)
    ffmpeg_dir_str = str(ffmpeg_dir)
    if ffmpeg_dir_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir_str + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

scripts_dir = str(Path(sys.prefix) / "Scripts")
if scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
