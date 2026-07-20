from io import BytesIO
from PIL import Image, UnidentifiedImageError
from app.core.exceptions import InvalidImageError

def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image") from exc
