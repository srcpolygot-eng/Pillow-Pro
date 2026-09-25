from __future__ import annotations

from PIL import Image
from PIL.engine.layers import Layer, LayerStack


class ImageLayers:
    """High-level layer API for Pillow-Pro."""

    def __init__(
        self,
        size: tuple[int, int],
        mode: str = "RGBA",
    ) -> None:
        width, height = size

        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")

        self.mode = mode
        self.stack = LayerStack(width, height)

    @property
    def size(self) -> tuple[int, int]:
        return (
            self.stack.width,
            self.stack.height,
        )

    @property
    def layers(self) -> list[Layer]:
        return self.stack.layers

    def add(
        self,
        image: Image.Image,
        name: str = "Layer",
        *,
        opacity: float = 1.0,
        visible: bool = True,
        x: int = 0,
        y: int = 0,
        blend_mode: str = "normal",
        locked: bool = False,
        index: int | None = None,
    ) -> Layer:
        """Add an image as a new layer."""

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        layer = Layer(
            image=image,
            name=name,
            opacity=opacity,
            visible=visible,
            x=x,
            y=y,
            blend_mode=blend_mode,
            locked=locked,
        )

        self.stack.add(layer, index)

        return layer

    def create(
        self,
        name: str = "Layer",
        *,
        opacity: float = 1.0,
        visible: bool = True,
        x: int = 0,
        y: int = 0,
        blend_mode: str = "normal",
        locked: bool = False,
        fill: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Layer:
        """Create a new blank layer."""

        image = Image.new(
            "RGBA",
            self.size,
            fill,
        )

        return self.add(
            image,
            name=name,
            opacity=opacity,
            visible=visible,
            x=x,
            y=y,
            blend_mode=blend_mode,
            locked=locked,
        )

    def remove(self, layer_or_index: Layer | int) -> Layer:
        """Remove a layer."""

        return self.stack.remove(layer_or_index)

    def duplicate(self, layer_or_index: Layer | int) -> Layer:
        """Duplicate a layer."""

        return self.stack.duplicate(layer_or_index)

    def move(
        self,
        layer_or_index: Layer | int,
        new_index: int,
    ) -> Layer:
        """Move a layer to another position in the stack."""

        return self.stack.move(
            layer_or_index,
            new_index,
        )

    def get(self, name: str) -> Layer | None:
        """Find a layer by name."""

        return self.stack.get(name)

    def clear(self) -> None:
        """Remove all layers."""

        self.stack.clear()

    def flatten(self) -> Image.Image:
        """Render all visible layers into one image."""

        return self.stack.flatten()

    def save(
        self,
        fp,
        format: str | None = None,
        **params,
    ) -> None:
        """Flatten and save the complete layer composition."""

        self.flatten().save(
            fp,
            format=format,
            **params,
        )


__all__ = [
    "ImageLayers",
]
