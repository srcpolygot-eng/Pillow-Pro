from __future__ import annotations

from colorsys import hls_to_rgb, rgb_to_hls
from enum import Enum
from typing import Iterable

from PIL import Image, ImageChops


_CHANNEL_MAX = 255.0
_EPSILON = 1e-12


class BlendMode(str, Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    ADD = "add"
    SUBTRACT = "subtract"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    DIVIDE = "divide"
    NEGATION = "negation"
    LINEAR_BURN = "linear_burn"
    VIVID_LIGHT = "vivid_light"
    PIN_LIGHT = "pin_light"
    HARD_MIX = "hard_mix"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    LUMINOSITY = "luminosity"


SUPPORTED_BLEND_MODES = frozenset(mode.value for mode in BlendMode)


class CompositingError(RuntimeError):
    """Base exception for compositing operations."""


class UnsupportedBlendModeError(CompositingError, ValueError):
    """Raised when an unknown blend mode is requested."""


def _ensure_image(image: Image.Image, name: str) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"{name} must be a PIL.Image.Image")
    return image


def _normalize_mode(mode: str | BlendMode) -> str:
    if isinstance(mode, BlendMode):
        return mode.value
    if not isinstance(mode, str):
        raise TypeError("blend mode must be a string or BlendMode")

    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "src_over": "normal",
        "source_over": "normal",
        "linear_dodge": "add",
        "hardlight": "hard_light",
        "softlight": "soft_light",
        "vividlight": "vivid_light",
        "pinlight": "pin_light",
        "hardmix": "hard_mix",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_BLEND_MODES:
        raise UnsupportedBlendModeError(
            f"Unsupported blend mode: {mode!r}. Supported modes: {', '.join(sorted(SUPPORTED_BLEND_MODES))}"
        )
    return normalized


def _validate_opacity(opacity: float) -> float:
    opacity = float(opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")
    return opacity


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _multiply(backdrop: float, source: float) -> float:
    return backdrop * source


def _screen(backdrop: float, source: float) -> float:
    return 1.0 - (1.0 - backdrop) * (1.0 - source)


def _overlay(backdrop: float, source: float) -> float:
    if backdrop <= 0.5:
        return 2.0 * backdrop * source
    return 1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source)


def _soft_light(backdrop: float, source: float) -> float:
    if source <= 0.5:
        return backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)

    if backdrop <= 0.25:
        d = ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop
    else:
        d = backdrop ** 0.5
    return backdrop + (2.0 * source - 1.0) * (d - backdrop)


def _hard_light(backdrop: float, source: float) -> float:
    if source <= 0.5:
        return 2.0 * backdrop * source
    return 1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source)


def _darken(backdrop: float, source: float) -> float:
    return min(backdrop, source)


def _lighten(backdrop: float, source: float) -> float:
    return max(backdrop, source)


def _difference(backdrop: float, source: float) -> float:
    return abs(backdrop - source)


def _exclusion(backdrop: float, source: float) -> float:
    return backdrop + source - 2.0 * backdrop * source


def _color_dodge(backdrop: float, source: float) -> float:
    if source >= 1.0 - _EPSILON:
        return 1.0
    return min(1.0, backdrop / max(_EPSILON, 1.0 - source))


def _color_burn(backdrop: float, source: float) -> float:
    if source <= _EPSILON:
        return 0.0
    return max(0.0, 1.0 - ((1.0 - backdrop) / max(source, _EPSILON)))


def _divide(backdrop: float, source: float) -> float:
    if source <= _EPSILON:
        return 1.0
    return min(1.0, backdrop / source)


def _negation(backdrop: float, source: float) -> float:
    return 1.0 - abs(1.0 - backdrop - source)


def _linear_burn(backdrop: float, source: float) -> float:
    return max(0.0, backdrop + source - 1.0)


def _vivid_light(backdrop: float, source: float) -> float:
    if source < 0.5:
        return _color_burn(backdrop, 2.0 * source)
    return _color_dodge(backdrop, 2.0 * (source - 0.5))


def _pin_light(backdrop: float, source: float) -> float:
    if source < 0.5:
        return min(backdrop, 2.0 * source)
    return max(backdrop, 2.0 * (source - 0.5))


def _hard_mix(backdrop: float, source: float) -> float:
    return 1.0 if _vivid_light(backdrop, source) >= 0.5 else 0.0


def _luminosity(rgb: tuple[float, float, float]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _set_luminosity(rgb: tuple[float, float, float], luminosity: float) -> tuple[float, float, float]:
    delta = luminosity - _luminosity(rgb)
    result = tuple(_clamp01(channel + delta) for channel in rgb)
    minimum = min(result)
    maximum = max(result)

    if minimum < 0.0:
        scale = luminosity / max(_EPSILON, luminosity - minimum)
        result = tuple(luminosity + (channel - luminosity) * scale for channel in result)
    if maximum > 1.0:
        scale = (1.0 - luminosity) / max(_EPSILON, maximum - luminosity)
        result = tuple(luminosity + (channel - luminosity) * scale for channel in result)
    return tuple(_clamp01(channel) for channel in result)


def _set_saturation(rgb: tuple[float, float, float], saturation: float) -> tuple[float, float, float]:
    minimum = min(rgb)
    maximum = max(rgb)
    if maximum <= minimum:
        return rgb
    scale = saturation / (maximum - minimum)
    return tuple(_clamp01((channel - minimum) * scale) for channel in rgb)


def _blend_channel(backdrop: float, source: float, mode: str) -> float:
    if mode == "normal":
        return source
    if mode == "multiply":
        return _multiply(backdrop, source)
    if mode == "screen":
        return _screen(backdrop, source)
    if mode == "overlay":
        return _overlay(backdrop, source)
    if mode == "soft_light":
        return _soft_light(backdrop, source)
    if mode == "hard_light":
        return _hard_light(backdrop, source)
    if mode == "darken":
        return _darken(backdrop, source)
    if mode == "lighten":
        return _lighten(backdrop, source)
    if mode == "difference":
        return _difference(backdrop, source)
    if mode == "exclusion":
        return _exclusion(backdrop, source)
    if mode == "add":
        return min(1.0, backdrop + source)
    if mode == "subtract":
        return max(0.0, backdrop - source)
    if mode == "color_dodge":
        return _color_dodge(backdrop, source)
    if mode == "color_burn":
        return _color_burn(backdrop, source)
    if mode == "divide":
        return _divide(backdrop, source)
    if mode == "negation":
        return _negation(backdrop, source)
    if mode == "linear_burn":
        return _linear_burn(backdrop, source)
    if mode == "vivid_light":
        return _vivid_light(backdrop, source)
    if mode == "pin_light":
        return _pin_light(backdrop, source)
    if mode == "hard_mix":
        return _hard_mix(backdrop, source)
    raise UnsupportedBlendModeError(f"Unsupported RGB blend mode: {mode!r}")


def _blend_hsl(backdrop: tuple[float, float, float], source: tuple[float, float, float], mode: str) -> tuple[float, float, float]:
    if mode == "hue":
        source_hue, _, source_lightness = rgb_to_hls(source[0], source[1], source[2])
        _, backdrop_saturation, _ = rgb_to_hls(backdrop[0], backdrop[1], backdrop[2])
        return tuple(_clamp01(channel) for channel in hls_to_rgb(source_hue, source_lightness, backdrop_saturation))

    if mode == "saturation":
        _, backdrop_lightness, _ = rgb_to_hls(backdrop[0], backdrop[1], backdrop[2])
        _, source_saturation, _ = rgb_to_hls(source[0], source[1], source[2])
        if backdrop_lightness >= 0:
            return _set_saturation(backdrop, source_saturation)
        return backdrop

    if mode == "color":
        source_hue, _, source_saturation = rgb_to_hls(source[0], source[1], source[2])
        _, backdrop_lightness, _ = rgb_to_hls(backdrop[0], backdrop[1], backdrop[2])
        return tuple(_clamp01(channel) for channel in hls_to_rgb(source_hue, backdrop_lightness, source_saturation))

    if mode == "luminosity":
        source_hue, source_lightness, source_saturation = rgb_to_hls(source[0], source[1], source[2])
        del source_hue, source_saturation
        return _set_luminosity(backdrop, source_lightness)

    raise UnsupportedBlendModeError(f"Unsupported HSL blend mode: {mode!r}")


def _fast_rgb_blend(backdrop: Image.Image, source: Image.Image, mode: str) -> Image.Image | None:
    if mode == "normal":
        return source.copy()
    if mode == "multiply":
        return ImageChops.multiply(backdrop, source)
    if mode == "screen":
        return ImageChops.screen(backdrop, source)
    if mode == "difference":
        return ImageChops.difference(backdrop, source)
    if mode == "darken":
        return ImageChops.darker(backdrop, source)
    if mode == "lighten":
        return ImageChops.lighter(backdrop, source)
    return None


def _blend_pixels(backdrop: Image.Image, source: Image.Image, mode: str) -> Image.Image:
    fast = _fast_rgb_blend(backdrop, source, mode)
    if fast is not None:
        return fast

    output: list[tuple[int, int, int]] = []
    hsl_mode = mode in {"hue", "saturation", "color", "luminosity"}
    for backdrop_pixel, source_pixel in zip(backdrop.getdata(), source.getdata()):
        backdrop_rgb = tuple(channel / _CHANNEL_MAX for channel in backdrop_pixel[:3])
        source_rgb = tuple(channel / _CHANNEL_MAX for channel in source_pixel[:3])

        if hsl_mode:
            blended = _blend_hsl(backdrop_rgb, source_rgb, mode)
        else:
            blended = tuple(_clamp01(_blend_channel(backdrop_rgb[index], source_rgb[index], mode)) for index in range(3))

        output.append(tuple(int(round(channel * _CHANNEL_MAX)) for channel in blended))

    return Image.new("RGB", backdrop.size, "black")._new(
        Image.core.new_block("RGB", backdrop.size[0], backdrop.size[1], bytes(value for pixel in output for value in pixel))
    )


def _normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL.Image.Image")
    if mask.size != size:
        raise ValueError("mask must have the same size as the images")
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask


def _apply_opacity_to_alpha(alpha: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return alpha
    if opacity <= 0.0:
        return Image.new("L", alpha.size, 0)
    return alpha.point(lambda value: int(round(value * opacity)))


def _combine_alpha_masks(first: Image.Image, second: Image.Image) -> Image.Image:
    return ImageChops.multiply(first, second)


def _source_over(backdrop: Image.Image, source: Image.Image, *, opacity: float, mask: Image.Image | None) -> Image.Image:
    backdrop = backdrop.convert("RGBA")
    source = source.convert("RGBA")
    source_alpha = source.getchannel("A")
    if opacity != 1.0:
        source_alpha = _apply_opacity_to_alpha(source_alpha, opacity)
    if mask is not None:
        mask = _normalize_mask(mask, source.size)
        source_alpha = _combine_alpha_masks(source_alpha, mask)
    source.putalpha(source_alpha)
    return Image.alpha_composite(backdrop, source)


def _blend_source_over(backdrop: Image.Image, source: Image.Image, *, mode: str, opacity: float, mask: Image.Image | None) -> Image.Image:
    backdrop = backdrop.convert("RGBA")
    source = source.convert("RGBA")
    source_alpha = source.getchannel("A")
    if opacity != 1.0:
        source_alpha = _apply_opacity_to_alpha(source_alpha, opacity)
    if mask is not None:
        mask = _normalize_mask(mask, source.size)
        source_alpha = _combine_alpha_masks(source_alpha, mask)

    backdrop_rgb = backdrop.convert("RGB")
    source_rgb = source.convert("RGB")
    blended_rgb = _blend_pixels(backdrop_rgb, source_rgb, mode)
    return _source_over(backdrop, blended_rgb.convert("RGBA"), opacity=1.0, mask=source_alpha)


def blend(base: Image.Image, layer: Image.Image, mode: str | BlendMode = BlendMode.NORMAL, opacity: float = 1.0, mask: Image.Image | None = None) -> Image.Image:
    _ensure_image(base, "base")
    _ensure_image(layer, "layer")
    if base.size != layer.size:
        raise ValueError("base and layer must have identical dimensions")

    opacity = _validate_opacity(opacity)
    mode = _normalize_mode(mode)

    if mode == "normal":
        return _source_over(base, layer.copy(), opacity=opacity, mask=mask)
    return _blend_source_over(base, layer.copy(), mode=mode, opacity=opacity, mask=mask)


def composite_at(base: Image.Image, layer: Image.Image, x: int, y: int, *, mode: str | BlendMode = BlendMode.NORMAL, opacity: float = 1.0, mask: Image.Image | None = None) -> Image.Image:
    _ensure_image(base, "base")
    _ensure_image(layer, "layer")
    opacity = _validate_opacity(opacity)
    mode = _normalize_mode(mode)

    base_rgba = base.convert("RGBA")
    layer_rgba = layer.convert("RGBA")

    left = int(x)
    top = int(y)
    right = left + layer_rgba.width
    bottom = top + layer_rgba.height

    clip_left = max(0, left)
    clip_top = max(0, top)
    clip_right = min(base_rgba.width, right)
    clip_bottom = min(base_rgba.height, bottom)

    if clip_left >= clip_right or clip_top >= clip_bottom:
        return base_rgba

    source_left = clip_left - left
    source_top = clip_top - top
    source_right = source_left + (clip_right - clip_left)
    source_bottom = source_top + (clip_bottom - clip_top)

    cropped_layer = layer_rgba.crop((source_left, source_top, source_right, source_bottom))
    cropped_mask = None
    if mask is not None:
        mask = _normalize_mask(mask, layer_rgba.size)
        cropped_mask = mask.crop((source_left, source_top, source_right, source_bottom))

    region = base_rgba.crop((clip_left, clip_top, clip_right, clip_bottom))
    result = blend(region, cropped_layer, mode=mode, opacity=opacity, mask=cropped_mask)
    output = base_rgba.copy()
    output.paste(result, (clip_left, clip_top))
    return output


def composite_many(base: Image.Image, layers: Iterable[Image.Image | tuple[Image.Image, str | BlendMode, float]]) -> Image.Image:
    result = base.convert("RGBA")
    for item in layers:
        if isinstance(item, Image.Image):
            image = item
            mode = BlendMode.NORMAL
            opacity = 1.0
        else:
            if len(item) != 3:
                raise ValueError("Layer tuples must contain (image, mode, opacity)")
            image, mode, opacity = item
        result = blend(result, image, mode=mode, opacity=opacity)
    return result


__all__ = [
    "BlendMode",
    "CompositingError",
    "SUPPORTED_BLEND_MODES",
    "UnsupportedBlendModeError",
    "blend",
    "composite_at",
    "composite_many",
]
