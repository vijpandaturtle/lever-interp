"""Synthetic image edits used as controlled stimuli.

Lives in the library (not inside one experiment) because typographic inputs are a recurring probe:
CLIP 03 (attack one image), CLIP 04 (attack rates across models), and later the driving stage
(sticker-on-sign attacks). One implementation keeps those results comparable.
"""

from PIL import Image, ImageDraw, ImageFont


def stamp_text(image: Image.Image, text: str, frac: float) -> Image.Image:
    """Render the attack text as a white-backed label below center, sized relative to the image.

    Programmatic rather than hand-edited so attacks are reproducible and sweepable (size, wording).
    The white box mimics the paper-note attack and keeps the text legible whatever the background.

    The font shrinks below `frac` when the label (plus box padding) would overflow the image width.
    Without this, long labels ("English springer" at 24px is ~185px) are clipped on small images, so
    some attack strings would be partially unreadable and attack rates would vary with label length.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    def load_font(size):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            return ImageFont.load_default(size)

    size = max(12, int(img.height * frac))
    font = load_font(size)
    while size > 8 and draw.textlength(text, font=font) + 16 > img.width:
        size -= 1
        font = load_font(size)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    pos = ((img.width - (x1 - x0)) // 2, int(img.height * 0.6))
    draw.rectangle((pos[0] - 8, pos[1] - 4, pos[0] + x1 - x0 + 8, pos[1] + y1 - y0 + 12), fill="white")
    draw.text(pos, text, fill="black", font=font)
    return img
