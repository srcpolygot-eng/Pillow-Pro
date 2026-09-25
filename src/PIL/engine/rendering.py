from __future__ import annotations

from PIL import Image

from .compositing import blend
from .layers import LayerStack


class Renderer:
    def __init__(self, stack: LayerStack) -> None:
        self.stack = stack

    def render(self) -> Image.Image:
        canvas = Image.new(
            "RGBA",
            (self.stack.width, self.stack.height),
            (0, 0, 0, 0),
        )

        for layer in self.stack.visible_layers():
            canvas = self._render_layer(canvas, layer)

        return canvas

    def _render_layer(
        self,
        canvas: Image.Image,
        layer,
    ) -> Image.Image:
        image = layer.image.convert("RGBA")

        if layer.opacity <= 0:
            return canvas

        if (
            layer.x == 0
            and layer.y == 0
            and image.size == canvas.size
        ):
            return blend(
                canvas,
                image,
                mode=layer.blend_mode,
                opacity=layer.opacity,
            )

        positioned = Image.new(
            "RGBA",
            canvas.size,
            (0, 0, 0, 0),
        )

        left = layer.x
        top = layer.y
        right = left + image.width
        bottom = top + image.height

        crop_left = max(0, -left)
        crop_top = max(0, -top)
        crop_right = min(image.width, canvas.width - left)
        crop_bottom = min(image.height, canvas.height - top)

        if crop_right <= crop_left or crop_bottom <= crop_top:
            return canvas

        cropped = image.crop(
            (
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            )
        )

        positioned.paste(
            cropped,
            (
                max(0, left),
                max(0, top),
            ),
            cropped,
        )

        return blend(
            canvas,
            positioned,
            mode=layer.blend_mode,
            opacity=layer.opacity,
        )


__all__ = [
    "Renderer",
]
