from __future__ import annotations

from PIL import Image, ImageChops


SUPPORTED_BLEND_MODES = {
    "normal",
    "multiply",
    "screen",
    "add",
    "subtract",
    "difference",
    "darken",
    "lighten",
}


def _ensure_rgba(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        return image

    return image.convert("RGBA")


def _apply_rgb_blend(
    base: Image.Image,
    layer: Image.Image,
    mode: str,
) -> Image.Image:
    base_rgb = base.convert("RGB")
    layer_rgb = layer.convert("RGB")

    if mode == "normal":
        return layer_rgb

    if mode == "multiply":
        return ImageChops.multiply(base_rgb, layer_rgb)

    if mode == "screen":
        return ImageChops.screen(base_rgb, layer_rgb)

    if mode == "add":
        return ImageChops.add(base_rgb, layer_rgb, scale=1.0, offset=0)

    if mode == "subtract":
        return ImageChops.subtract(base_rgb, layer_rgb, scale=1.0, offset=0)

    if mode == "difference":
        return ImageChops.difference(base_rgb, layer_rgb)

    if mode == "darken":
        return ImageChops.darker(base_rgb, layer_rgb)

    if mode == "lighten":
        return ImageChops.lighter(base_rgb, layer_rgb)

    raise ValueError(f"Unsupported blend mode: {mode}")


def blend(
    base: Image.Image,
    layer: Image.Image,
    mode: str = "normal",
    opacity: float = 1.0,
) -> Image.Image:
    if not isinstance(base, Image.Image):
        raise TypeError("base must be a PIL.Image.Image")

    if not isinstance(layer, Image.Image):
        raise TypeError("layer must be a PIL.Image.Image")

    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")

    mode = mode.lower()

    if mode not in SUPPORTED_BLEND_MODES:
        raise ValueError(
            f"Unsupported blend mode: {mode}. "
            f"Supported modes: {', '.join(sorted(SUPPORTED_BLEND_MODES))}"
        )

    base_rgba = _ensure_rgba(base)
    layer_rgba = _ensure_rgba(layer)

    if base_rgba.size != layer_rgba.size:
        raise ValueError("Images must have identical dimensions")

    blended_rgb = _apply_rgb_blend(
        base_rgba,
        layer_rgba,
        mode,
    ).convert("RGBA")

    layer_alpha = layer_rgba.getchannel("A")

    if opacity != 1.0:
        layer_alpha = layer_alpha.point(
            lambda value: int(value * opacity)
        )

    blended_rgb.putalpha(layer_alpha)

    return Image.alpha_composite(
        base_rgba,
        blended_rgb,
    )


__all__ = [
    "SUPPORTED_BLEND_MODES",
    "blend",
]
