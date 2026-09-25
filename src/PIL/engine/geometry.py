from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def translated(self, dx: float, dy: float) -> Point:
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True, slots=True)
class Size:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Size dimensions cannot be negative")


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def size(self) -> Size:
        return Size(
            max(0, int(self.width)),
            max(0, int(self.height)),
        )

    def contains(self, point: Point) -> bool:
        return (
            self.left <= point.x < self.right
            and self.top <= point.y < self.bottom
        )

    def intersects(self, other: Rect) -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )

    def intersection(self, other: Rect) -> Rect | None:
        if not self.intersects(other):
            return None

        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return Rect(
            left,
            top,
            right - left,
            bottom - top,
        )

    def translated(self, dx: float, dy: float) -> Rect:
        return Rect(
            self.x + dx,
            self.y + dy,
            self.width,
            self.height,
        )


__all__ = [
    "Point",
    "Rect",
    "Size",
]
