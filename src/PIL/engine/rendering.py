from __future__ import annotations

import math
from typing import Iterable, Iterator

from PIL import Image, ImageChops

from .compositing import BlendMode, blend, composite_at
from .layers import Layer, LayerGroup, LayerStack


class RenderError(RuntimeError):
    """Base exception for rendering failures."""


class RenderSizeError(RenderError):
    """Raised when an image cannot be rendered into the target size."""


class RenderOptions:
    """Configuration for a rendering operation."""

    def __init__(
        self,
        *,
        mode: str = "RGBA",
        background: tuple[int, int, int, int] | None = None,
        include_hidden: bool = False,
        include_zero_opacity: bool = False,
        apply_masks: bool = True,
        apply_clipping: bool = True,
        clear_transparent: bool = False,
    ) -> None:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if background is not None:
            if len(background) != 4:
                raise ValueError("background must contain four channels")
            for value in background:
                if not 0 <= int(value) <= 255:
                    raise ValueError("background channels must be between 0 and 255")

        self.mode = mode
        self.background = background
        self.include_hidden = include_hidden
        self.include_zero_opacity = include_zero_opacity
        self.apply_masks = apply_masks
        self.apply_clipping = apply_clipping
        self.clear_transparent = clear_transparent


class RenderResult:
    """Result of a rendering operation."""

    def __init__(self, image: Image.Image, *, layers_rendered: int = 0, layers_skipped: int = 0, groups_rendered: int = 0) -> None:
        self.image = image
        self.layers_rendered = layers_rendered
        self.layers_skipped = layers_skipped
        self.groups_rendered = groups_rendered

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def mode(self) -> str:
        return self.image.mode


class Renderer:
    """Render a LayerStack into a Pillow image."""

    def __init__(self, stack: LayerStack, options: RenderOptions | None = None) -> None:
        if not isinstance(stack, LayerStack):
            raise TypeError("stack must be a LayerStack")
        self.stack = stack
        self.options = options or RenderOptions()
        self._layers_rendered = 0
        self._layers_skipped = 0
        self._groups_rendered = 0

    def render(self) -> Image.Image:
        return self.render_result().image

    def render_result(self) -> RenderResult:
        self._reset_statistics()
        canvas = self._create_canvas()

        for layer in self.stack:
            canvas = self._render_layer(canvas, layer)

        canvas = self._finalize(canvas)
        return RenderResult(
            image=canvas,
            layers_rendered=self._layers_rendered,
            layers_skipped=self._layers_skipped,
            groups_rendered=self._groups_rendered,
        )

    def _create_canvas(self) -> Image.Image:
        size = self.stack.size
        if self.options.background is None:
            return Image.new("RGBA", size, (0, 0, 0, 0))
        return Image.new("RGBA", size, self.options.background)

    def _finalize(self, canvas: Image.Image) -> Image.Image:
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")

        if self.options.clear_transparent:
            canvas = self._clear_transparent_rgb(canvas)

        if self.options.mode != "RGBA":
            try:
                canvas = canvas.convert(self.options.mode)
            except Exception as exc:
                raise RenderError(f"Unable to convert rendered image to mode {self.options.mode!r}") from exc
        return canvas

    def _clear_transparent_rgb(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb = rgba.convert("RGB")
        if alpha.mode != "L":
            alpha = alpha.convert("L")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        result = Image.composite(background, rgb, alpha)
        return result.convert("RGBA")

    def _render_layer(self, canvas: Image.Image, layer: Layer) -> Image.Image:
        if not self._should_render(layer):
            self._layers_skipped += 1
            return canvas

        if isinstance(layer, LayerGroup):
            return self._render_group(canvas, layer)
        return self._render_single_layer(canvas, layer)

    def _should_render(self, layer: Layer) -> bool:
        if not self.options.include_hidden and not layer.visible:
            return False
        if not self.options.include_zero_opacity and layer.opacity <= 0.0:
            return False
        return True

    def _render_single_layer(self, canvas: Image.Image, layer: Layer) -> Image.Image:
        image = self._prepare_layer_image(layer)
        if image.width == 0 or image.height == 0:
            self._layers_skipped += 1
            return canvas

        positioned = self._position_layer(image, layer.x, layer.y, canvas.size)
        if positioned is None:
            self._layers_skipped += 1
            return canvas

        if layer.mask is not None and self.options.apply_masks:
            positioned = self._apply_layer_mask(positioned, layer.mask, layer.x, layer.y, canvas.size)

        canvas = blend(canvas, positioned, mode=layer.blend_mode, opacity=layer.opacity)
        self._layers_rendered += 1
        return canvas

    def _render_group(self, canvas: Image.Image, group: LayerGroup) -> Image.Image:
        group_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        for child in group.children:
            group_canvas = self._render_layer(group_canvas, child)

        if group.mask is not None and self.options.apply_masks:
            group_canvas = self._apply_layer_mask(group_canvas, group.mask, group.x, group.y, canvas.size)

        if group.opacity <= 0.0:
            self._layers_skipped += 1
            return canvas

        result = blend(canvas, group_canvas, mode=group.blend_mode, opacity=group.opacity)
        self._groups_rendered += 1
        self._layers_rendered += 1
        return result

    def _prepare_layer_image(self, layer: Layer) -> Image.Image:
        image = layer.image
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        else:
            image = image.copy()
        return image

    def _position_layer(self, image: Image.Image, x: float, y: float, canvas_size: tuple[int, int]) -> Image.Image | None:
        canvas_width, canvas_height = canvas_size
        left = int(round(x))
        top = int(round(y))
        right = left + image.width
        bottom = top + image.height

        clipped_left = max(0, left)
        clipped_top = max(0, top)
        clipped_right = min(canvas_width, right)
        clipped_bottom = min(canvas_height, bottom)

        if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
            return None

        source_left = clipped_left - left
        source_top = clipped_top - top
        source_right = source_left + (clipped_right - clipped_left)
        source_bottom = source_top + (clipped_bottom - clipped_top)

        cropped = image.crop((source_left, source_top, source_right, source_bottom))
        positioned = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        positioned.alpha_composite(cropped, (clipped_left, clipped_top))
        return positioned

    def _apply_layer_mask(self, image: Image.Image, mask: Image.Image, x: float, y: float, canvas_size: tuple[int, int]) -> Image.Image:
        mask = self._prepare_mask(mask, x, y, canvas_size)
        alpha = image.getchannel("A")
        alpha = ImageChops.multiply(alpha, mask)
        result = image.copy()
        result.putalpha(alpha)
        return result

    def _prepare_mask(self, mask: Image.Image, x: float, y: float, canvas_size: tuple[int, int]) -> Image.Image:
        if mask.mode != "L":
            if "A" in mask.getbands():
                mask = mask.getchannel("A")
            else:
                mask = mask.convert("L")
        else:
            mask = mask.copy()

        positioned = self._position_mask(mask, x, y, canvas_size)
        if positioned is None:
            return Image.new("L", canvas_size, 0)
        return positioned

    def _position_mask(self, mask: Image.Image, x: float, y: float, canvas_size: tuple[int, int]) -> Image.Image | None:
        canvas_width, canvas_height = canvas_size
        left = int(round(x))
        top = int(round(y))
        right = left + mask.width
        bottom = top + mask.height

        clipped_left = max(0, left)
        clipped_top = max(0, top)
        clipped_right = min(canvas_width, right)
        clipped_bottom = min(canvas_height, bottom)
        if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
            return None

        source_left = clipped_left - left
        source_top = clipped_top - top
        source_right = source_left + (clipped_right - clipped_left)
        source_bottom = source_top + (clipped_bottom - clipped_top)

        cropped = mask.crop((source_left, source_top, source_right, source_bottom))
        positioned = Image.new("L", canvas_size, 0)
        positioned.paste(cropped, (clipped_left, clipped_top))
        return positioned

    def render_clipped(self, rect) -> Image.Image:
        full = self.render()
        left = max(0, int(round(rect.x)))
        top = max(0, int(round(rect.y)))
        right = min(full.width, int(round(rect.x + rect.width)))
        bottom = min(full.height, int(round(rect.y + rect.height)))

        if right <= left or bottom <= top:
            return Image.new("RGBA", (0, 0))
        return full.crop((left, top, right, bottom))

    def render_region(self, box: tuple[int, int, int, int]) -> Image.Image:
        if len(box) != 4:
            raise ValueError("box must contain four values")
        left, top, right, bottom = (int(value) for value in box)
        left = max(0, left)
        top = max(0, top)
        right = min(self.stack.width, right)
        bottom = min(self.stack.height, bottom)
        if right <= left or bottom <= top:
            return Image.new("RGBA", (0, 0))
        return self.render().crop((left, top, right, bottom))

    def thumbnail(self, size: tuple[int, int], *, resample: int = Image.Resampling.LANCZOS) -> Image.Image:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError("thumbnail size must contain positive dimensions")

        image = self.render().copy()
        image.thumbnail(size, resample=resample)
        return image

    def _reset_statistics(self) -> None:
        self._layers_rendered = 0
        self._layers_skipped = 0
        self._groups_rendered = 0


def render(stack: LayerStack, *, mode: str = "RGBA", background: tuple[int, int, int, int] | None = None) -> Image.Image:
    renderer = Renderer(stack, RenderOptions(mode=mode, background=background))
    return renderer.render()


def render_layers(layers: Iterable[Layer], size: tuple[int, int], *, mode: str = "RGBA") -> Image.Image:
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")

    stack = LayerStack(width, height)
    for layer in layers:
        stack.add(layer)
    return render(stack, mode=mode)


__all__ = [
    "RenderError",
    "RenderOptions",
    "RenderResult",
    "RenderSizeError",
    "Renderer",
    "render",
    "render_layers",
]
