# smart_icon_downscale_rgba.py
import os
import sys

from PIL import Image, ImageFilter, ImageOps


def _resampling():
    # Pillow >=9.1 uses Image.Resampling
    return getattr(Image, "Resampling", Image).Resampling.LANCZOS


def _autocontrast_rgba(img_rgba, cutoff=1):
    r, g, b, a = img_rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = ImageOps.autocontrast(rgb, cutoff=cutoff)
    r2, g2, b2 = rgb.split()
    out = Image.merge("RGBA", (r2, g2, b2, a))
    return out


def downscale_pixel_art(img, size):
    return img.resize(size, Image.Resampling.NEAREST)


def downscale_photo(img, size):
    img = ImageOps.exif_transpose(img).convert("RGBA")
    lanczos = _resampling()

    # Progressive downscale to avoid aliasing on big shrinks
    w, h = img.size
    tw, th = size
    while max(w, h) > max(tw * 2, th * 2):
        w = max(tw * 2, w // 2)
        h = max(th * 2, h // 2)
        img = img.resize((w, h), lanczos)

    img = img.resize(size, lanczos)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.6, percent=140, threshold=2))
    img = _autocontrast_rgba(img, cutoff=1)  # <-- safe for RGBA now
    return img


def convert(input_path, output_path=None, size=(16, 16), mode="auto"):
    img = Image.open(input_path).convert("RGBA")
    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_{size[0]}x{size[1]}.png"

    if mode == "pixel":
        out = downscale_pixel_art(img, size)
    elif mode == "photo":
        out = downscale_photo(img, size)
    else:
        w, h = img.size
        if w % size[0] == 0 and h % size[1] == 0:
            out = downscale_pixel_art(img, size)
        else:
            out = downscale_photo(img, size)

    out.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python smart_icon_downscale_rgba.py <input> [output] [mode=auto|pixel|photo]"
        )
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    mode = sys.argv[3] if len(sys.argv) >= 4 else "auto"
    convert(inp, out, (16, 16), mode)
