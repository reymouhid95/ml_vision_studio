from PIL import Image, ImageEnhance, ImageOps


def augment_image(img: Image.Image) -> list[Image.Image]:
    """Return [flipped, brightened] — both 224x224, same as the JS augmentImg()."""
    img = img.resize((224, 224)).convert("RGB")
    flipped = ImageOps.mirror(img)
    brightened = ImageEnhance.Brightness(img).enhance(1.3)
    return [flipped, brightened]
