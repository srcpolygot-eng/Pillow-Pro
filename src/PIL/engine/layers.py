from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from PIL import Image


@dataclass
class Layer:
    image: Image.Image
    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    x: int = 0
    y: int = 0
    blend_mode: str = "normal"
    locked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        self.opacity = self._validate_opacity(self.opacity)
        self.blend_mode = self.blend_mode.lower()

    @staticmethod
    def _validate_opacity(value: float) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError("opacity must be between 0.0 and 1.0")

        return value

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.x + self.image.width,
            self.y + self.image.height,
        )

    def copy(self) -> Layer:
        return Layer(
            image=self.image.copy(),
            name=self.name,
            opacity=self.opacity,
            visible=self.visible,
            x=self.x,
            y=self.y,
            blend_mode=self.blend_mode,
            locked=self.locked,
        )

    def move(self, x: int, y: int) -> Layer:
        if self.locked:
            raise RuntimeError("Layer is locked")

        self.x = int(x)
        self.y = int(y)

        return self

    def translate(self, dx: int, dy: int) -> Layer:
        if self.locked:
            raise RuntimeError("Layer is locked")

        self.x += int(dx)
        self.y += int(dy)

        return self

    def set_opacity(self, opacity: float) -> Layer:
        if self.locked:
            raise RuntimeError("Layer is locked")

        self.opacity = self._validate_opacity(opacity)

        return self

    def set_visible(self, visible: bool) -> Layer:
        if self.locked:
            raise RuntimeError("Layer is locked")

        self.visible = bool(visible)

        return self


@dataclass
class LayerStack:
    width: int
    height: int
    layers: list[Layer] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("LayerStack dimensions must be positive")

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def add(
        self,
        layer: Layer,
        index: int | None = None,
    ) -> Layer:
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")

        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)

        return layer

    def create(
        self,
        image: Image.Image,
        name: str = "Layer",
        **kwargs,
    ) -> Layer:
        layer = Layer(
            image=image,
            name=name,
            **kwargs,
        )

        self.add(layer)

        return layer

    def remove(self, layer_or_index: Layer | int) -> Layer:
        if isinstance(layer_or_index, int):
            return self.layers.pop(layer_or_index)

        self.layers.remove(layer_or_index)

        return layer_or_index

    def clear(self) -> None:
        self.layers.clear()

    def move(
        self,
        layer_or_index: Layer | int,
        new_index: int,
    ) -> Layer:
        if isinstance(layer_or_index, int):
            layer = self.layers.pop(layer_or_index)
        else:
            layer = layer_or_index
            self.layers.remove(layer)

        self.layers.insert(new_index, layer)

        return layer

    def duplicate(self, layer_or_index: Layer | int) -> Layer:
        layer = (
            self.layers[layer_or_index]
            if isinstance(layer_or_index, int)
            else layer_or_index
        )

        copy = layer.copy()
        copy.name = f"{layer.name} copy"

        index = self.layers.index(layer)
        self.layers.insert(index + 1, copy)

        return copy

    def get(self, name: str) -> Layer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer

        return None

    def visible_layers(self) -> Iterable[Layer]:
        return (
            layer
            for layer in self.layers
            if layer.visible and layer.opacity > 0
        )

    def flatten(self) -> Image.Image:
        from .rendering import Renderer

        return Renderer(self).render()


__all__ = [
    "Layer",
    "LayerStack",
]
