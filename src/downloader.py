# src/downloader.py

import os
from pathlib import Path
import ffmpeg
from yt_dlp import YoutubeDL
from logger import logger

def download_and_transcode(url: str,
                           output_dir: str = "videos",
                           audio_bitrate: str = "192k") -> str:
    """
    1) Download bestvideo+bestaudio via yt-dlp → initial .mp4 (may contain Opus audio).
    2) Transcode audio to AAC (copying video) so that the file is playable on Windows.

    Returns the path to the final, AAC-audio MP4.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Step A: download & merge ---
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "noplaylist": True,
        "quiet": False,
        "nopart": True,             # don't use .part temp files
        "windowsfilenames": True,    # sanitize names for Windows
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path( ydl.prepare_filename(info) ).with_suffix(".mp4")
    except Exception as e:
        print(f"Failed to download video: {e}")
        logger.info(f"Failed to download video: {e}")
        raise

    # --- Step B: transcode audio to AAC ---
    final_name = downloaded.stem + "_aac.mp4"
    final_path = downloaded.with_name(final_name)

    try:
        (
            ffmpeg
            .input(str(downloaded))
            .output(
                str(final_path),
                vcodec="copy",          # keep original video
                acodec="aac",           # re-encode audio to AAC
                audio_bitrate=audio_bitrate
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as err:
        print(f"Error transcoding audio: {err.stderr.decode()}")
        logger.info(f"Error transcoding audio: {err.stderr.decode()}")
        raise

    # remove the original opus-audio file to save space
    try:
        downloaded.unlink()
    except OSError:
        pass

    return str(final_path)
