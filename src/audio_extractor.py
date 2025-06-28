import os
from pathlib import Path
import ffmpeg

def extract_audio(video_path: str,
                  output_format: str = "mp3",
                  output_dir: str = "audio") -> str:
    """
    Extracts and converts the audio track from a video file.

    Args:
        video_path (str): Path to input MP4 (with AAC audio).
        output_format (str): "mp3" or "wav".
        output_dir (str): Directory to save extracted audio.

    Returns:
        str: Path to the extracted audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(video_path).stem
    out_path = Path(output_dir) / f"{base}.{output_format}"

    # Configure codec based on desired format
    if output_format == "mp3":
        codec = "libmp3lame"
        codec_args = {"ar": "16000", "ac": "1", "b:a": "192k"}
    else:
        codec = "pcm_s16le"
        codec_args = {"ar": "16000", "ac": "1"}

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(out_path),
                    vn=None,            # drop video
                    acodec=codec,
                    **codec_args)
            .overwrite_output()
            .run(quiet=True)
        )
        return str(out_path)
    except ffmpeg.Error as e:
        err = e.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{err}")
