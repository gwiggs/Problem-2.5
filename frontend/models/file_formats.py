from enum import Enum

class VideoFormat(str, Enum):
    """Supported video file formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    MKV = "mkv"
    FLV = "flv"
    WMV = "wmv"

class ImageFormat(str, Enum):
    """Supported image file formats."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"
    SVG = "svg"

# Helper functions to get all formats
def get_all_video_formats() -> list[str]:
    """Get all supported video formats as a list of strings."""
    return [format.value for format in VideoFormat]

def get_all_image_formats() -> list[str]:
    """Get all supported image formats as a list of strings."""
    return [format.value for format in ImageFormat]

def is_video_format(filename: str) -> bool:
    """Check if a filename has a video format extension."""
    return any(filename.lower().endswith(f".{fmt}") for fmt in get_all_video_formats())

def is_image_format(filename: str) -> bool:
    """Check if a filename has an image format extension."""
    return any(filename.lower().endswith(f".{fmt}") for fmt in get_all_image_formats()) 