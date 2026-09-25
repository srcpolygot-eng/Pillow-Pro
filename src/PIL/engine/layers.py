from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator
from uuid import UUID, uuid4

from PIL import Image

from .geometry import Point, Rect, Size

# ------ Layer Errors ------ #


class LayerError(RuntimeError):
    """Base exception for layer-system errors."""


class LayerLockedError(LayerError):
    """Raised when attempting to modify a locked layer."""


class LayerNotFoundError(LayerError):
    """Raised when a requested layer does not exist."""


# ------ Layer Constants ------ #


SUPPORTED_BLEND_MODES = frozenset(
    {
        "normal",
        "multiply",
        "screen",
        "overlay",
        "soft_light",
        "hard_light",
        "darken",
        "lighten",
        "difference",
        "exclusion",
        "add",
        "subtract",
        "color_dodge",
        "color_burn",
        "divide",
        "negation",
        "linear_burn",
        "vivid_light",
        "pin_light",
        "hard_mix",
        "hue",
        "saturation",
        "color",
        "luminosity",
    }
)


# ------ Layer ------ #


@dataclass
class Layer:
    """A single editable image layer."""

    image: Image.Image
    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    x: float = 0.0
    y: float = 0.0
    blend_mode: str = "normal"
    locked: bool = False
    clipping: bool = False
    mask: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    id: UUID = field(default_factory=uuid4, init=False)
    parent: Layer | None = field(default=None, init=False, repr=False, compare=False)
    _dirty: bool = field(default=True, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        self.opacity = self._validate_opacity(self.opacity)
        self.blend_mode = self._validate_blend_mode(self.blend_mode)

        if self.mask is not None:
            self._validate_mask(self.mask)

    @staticmethod
    def _validate_opacity(opacity: float) -> float:
        opacity = float(opacity)
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between 0.0 and 1.0")
        return opacity

    @staticmethod
    def _validate_blend_mode(blend_mode: str) -> str:
        if not isinstance(blend_mode, str):
            raise TypeError("blend_mode must be a string")
        normalized = blend_mode.lower().strip().replace("-", "_").replace(" ", "_")
        if normalized not in SUPPORTED_BLEND_MODES:
            raise ValueError(f"Unsupported blend mode: {blend_mode!r}")
        return normalized

    def _validate_mask(self, mask: Image.Image) -> None:
        if not isinstance(mask, Image.Image):
            raise TypeError("mask must be a PIL.Image.Image")
        if mask.size != self.image.size:
            raise ValueError("mask must have the same size as the layer")

    @property
    def size(self) -> Size:
        return Size(self.image.width, self.image.height)

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    @property
    def bounds(self) -> Rect:
        return Rect(self.x, self.y, self.image.width, self.image.height)

    @property
    def center(self) -> Point:
        return self.bounds.center

    @property
    def area(self) -> int:
        return self.image.width * self.image.height

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def is_group(self) -> bool:
        return False

    def mark_dirty(self) -> None:
        self._dirty = True
        if self.parent is not None:
            self.parent.mark_dirty()

    def clear_dirty(self) -> None:
        self._dirty = False

    def ensure_unlocked(self) -> None:
        if self.locked:
            raise LayerLockedError(f"Layer {self.name!r} is locked")

    def move(self, x: float, y: float) -> Layer:
        self.ensure_unlocked()
        self.x = float(x)
        self.y = float(y)
        self.mark_dirty()
        return self

    def translate(self, dx: float, dy: float) -> Layer:
        self.ensure_unlocked()
        self.x += float(dx)
        self.y += float(dy)
        self.mark_dirty()
        return self

    def set_opacity(self, opacity: float) -> Layer:
        self.ensure_unlocked()
        self.opacity = self._validate_opacity(opacity)
        self.mark_dirty()
        return self

    def set_visible(self, visible: bool) -> Layer:
        self.visible = bool(visible)
        self.mark_dirty()
        return self

    def set_blend_mode(self, blend_mode: str) -> Layer:
        self.ensure_unlocked()
        self.blend_mode = self._validate_blend_mode(blend_mode)
        self.mark_dirty()
        return self

    def set_locked(self, locked: bool) -> Layer:
        self.locked = bool(locked)
        self.mark_dirty()
        return self

    def replace_image(self, image: Image.Image) -> Layer:
        self.ensure_unlocked()
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        self.image = image
        if self.mask is not None and self.mask.size != image.size:
            self.mask = None
        self.mark_dirty()
        return self

    def convert(self, mode: str) -> Layer:
        self.ensure_unlocked()
        self.image = self.image.convert(mode)
        self.mark_dirty()
        return self

    def set_mask(self, mask: Image.Image | None) -> Layer:
        self.ensure_unlocked()
        if mask is not None:
            self._validate_mask(mask)
        self.mask = mask
        self.mark_dirty()
        return self

    def remove_mask(self) -> Image.Image | None:
        self.ensure_unlocked()
        mask = self.mask
        self.mask = None
        self.mark_dirty()
        return mask

    def set_metadata(self, key: str, value: Any) -> Layer:
        self.metadata[key] = value
        self.mark_dirty()
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def remove_metadata(self, key: str) -> Any:
        value = self.metadata.pop(key, None)
        self.mark_dirty()
        return value

    def copy(self, *, name: str | None = None, copy_metadata: bool = True) -> Layer:
        duplicate = Layer(
            image=self.image.copy(),
            name=self.name if name is None else name,
            opacity=self.opacity,
            visible=self.visible,
            x=self.x,
            y=self.y,
            blend_mode=self.blend_mode,
            locked=self.locked,
            clipping=self.clipping,
            mask=self.mask.copy() if self.mask is not None else None,
            metadata=dict(self.metadata) if copy_metadata else {},
        )
        return duplicate

    def __repr__(self) -> str:
        return (
            f"Layer(name={self.name!r}, "
            f"size={self.image.size!r}, "
            f"position=({self.x}, {self.y}), "
            f"opacity={self.opacity}, "
            f"visible={self.visible}, "
            f"blend_mode={self.blend_mode!r}, "
            f"id={str(self.id)[:8]!r})"
        )


@dataclass
class LayerGroup(Layer):
    """A container layer holding child layers."""

    children: list[Layer] = field(default_factory=list)
    expanded: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for child in self.children:
            child.parent = self

    @property
    def is_group(self) -> bool:
        return True

    def add(self, layer: Layer, index: int | None = None) -> Layer:
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        if layer is self:
            raise ValueError("A group cannot contain itself")
        if self._is_ancestor_of(layer):
            raise ValueError("Cannot create a cyclic layer hierarchy")
        if layer.parent is not None:
            layer.parent.remove(layer)

        layer.parent = self
        if index is None:
            self.children.append(layer)
        else:
            self.children.insert(index, layer)
        self.mark_dirty()
        return layer

    def remove(self, layer_or_index: Layer | int) -> Layer:
        if isinstance(layer_or_index, int):
            layer = self.children.pop(layer_or_index)
        else:
            layer = layer_or_index
            self.children.remove(layer)

        layer.parent = None
        self.mark_dirty()
        return layer

    def _is_ancestor_of(self, layer: Layer) -> bool:
        current = self.parent
        while current is not None:
            if current is layer:
                return True
            current = current.parent
        return False

    def find(self, name: str) -> Layer | None:
        for child in self.children:
            if child.name == name:
                return child
            if isinstance(child, LayerGroup):
                result = child.find(name)
                if result is not None:
                    return result
        return None

    def walk(self) -> Iterator[Layer]:
        for child in self.children:
            yield child
            if isinstance(child, LayerGroup):
                yield from child.walk()

    def duplicate(self, *, name: str | None = None) -> LayerGroup:
        duplicate = LayerGroup(
            image=self.image.copy(),
            name=self.name if name is None else name,
            opacity=self.opacity,
            visible=self.visible,
            x=self.x,
            y=self.y,
            blend_mode=self.blend_mode,
            locked=self.locked,
            clipping=self.clipping,
            mask=self.mask.copy() if self.mask is not None else None,
            metadata=dict(self.metadata),
            expanded=self.expanded,
        )
        for child in self.children:
            duplicate.add(child.duplicate() if isinstance(child, LayerGroup) else child.copy())
        return duplicate


class LayerStack:
    """Root layer tree for an editable Pillow-Pro document."""

    def __init__(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("LayerStack dimensions must be positive")

        self._size = Size(width, height)
        self._layers: list[Layer] = []
        self._dirty = True

    @property
    def width(self) -> int:
        return self._size.width

    @property
    def height(self) -> int:
        return self._size.height

    @property
    def size(self) -> tuple[int, int]:
        return self._size.as_tuple()

    @property
    def layers(self) -> list[Layer]:
        return list(self._layers)

    @property
    def count(self) -> int:
        return len(self._layers)

    @property
    def is_dirty(self) -> bool:
        return self._dirty or any(layer.is_dirty for layer in self.walk())

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers)

    def __getitem__(self, index: int) -> Layer:
        return self._layers[index]

    def mark_dirty(self) -> None:
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False
        for layer in self.walk():
            layer.clear_dirty()

    def add(self, layer: Layer, index: int | None = None) -> Layer:
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        if layer.parent is not None:
            layer.parent.remove(layer)

        layer.parent = None
        if index is None:
            self._layers.append(layer)
        else:
            if index < 0:
                index = max(0, len(self._layers) + index)
            index = min(index, len(self._layers))
            self._layers.insert(index, layer)
        self.mark_dirty()
        return layer

    def create(self, image: Image.Image, name: str = "Layer", **kwargs: Any) -> Layer:
        layer = Layer(image=image, name=name, **kwargs)
        return self.add(layer)

    def create_group(self, name: str = "Group", **kwargs: Any) -> LayerGroup:
        group = LayerGroup(
            image=Image.new("RGBA", self.size, (0, 0, 0, 0)),
            name=name,
            **kwargs,
        )
        return self.add(group)

    def remove(self, layer_or_index: Layer | int) -> Layer:
        if isinstance(layer_or_index, int):
            layer = self._layers.pop(layer_or_index)
        else:
            layer = layer_or_index
            if layer.parent is not None:
                return layer.parent.remove(layer)
            self._layers.remove(layer)

        layer.parent = None
        self.mark_dirty()
        return layer

    def clear(self) -> None:
        for layer in self._layers:
            layer.parent = None
        self._layers.clear()
        self.mark_dirty()

    def move(self, layer_or_index: Layer | int, new_index: int) -> Layer:
        if isinstance(layer_or_index, int):
            layer = self._layers.pop(layer_or_index)
        else:
            layer = layer_or_index
            if layer.parent is not None:
                raise LayerError("Nested layers must be moved through their parent group")
            self._layers.remove(layer)

        new_index = max(0, min(int(new_index), len(self._layers)))
        self._layers.insert(new_index, layer)
        self.mark_dirty()
        return layer

    def raise_layer(self, layer: Layer) -> Layer:
        index = self._layers.index(layer)
        if index < len(self._layers) - 1:
            self.move(layer, index + 1)
        return layer

    def lower_layer(self, layer: Layer) -> Layer:
        index = self._layers.index(layer)
        if index > 0:
            self.move(layer, index - 1)
        return layer

    def move_to_top(self, layer: Layer) -> Layer:
        return self.move(layer, len(self._layers))

    def move_to_bottom(self, layer: Layer) -> Layer:
        return self.move(layer, 0)

    def find(self, name: str) -> Layer | None:
        for layer in self._layers:
            if layer.name == name:
                return layer
            if isinstance(layer, LayerGroup):
                result = layer.find(name)
                if result is not None:
                    return result
        return None

    def get(self, name: str) -> Layer | None:
        return self.find(name)

    def find_by_id(self, layer_id: UUID) -> Layer | None:
        for layer in self.walk():
            if layer.id == layer_id:
                return layer
        return None

    def walk(self) -> Iterator[Layer]:
        for layer in self._layers:
            yield layer
            if isinstance(layer, LayerGroup):
                yield from layer.walk()

    def visible_layers(self) -> Iterable[Layer]:
        return (layer for layer in self.walk() if layer.visible and layer.opacity > 0.0)

    def duplicate(self, layer_or_index: Layer | int) -> Layer:
        if isinstance(layer_or_index, int):
            layer = self._layers[layer_or_index]
        else:
            layer = layer_or_index

        duplicate = layer.duplicate() if isinstance(layer, LayerGroup) else layer.copy()
        if layer.parent is not None:
            parent = layer.parent
            index = parent.children.index(layer)
            parent.add(duplicate, index + 1)
        else:
            index = self._layers.index(layer)
            self.add(duplicate, index + 1)
        return duplicate

    def flatten(self) -> Image.Image:
        from .rendering import Renderer

        return Renderer(self).render()

    def __repr__(self) -> str:
        return f"LayerStack(size={self.size!r}, layers={len(self._layers)})"


__all__ = [
    "SUPPORTED_BLEND_MODES",
    "Layer",
    "LayerError",
    "LayerGroup",
    "LayerLockedError",
    "LayerNotFoundError",
    "LayerStack",
]
