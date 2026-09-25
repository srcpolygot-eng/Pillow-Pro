from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Iterator


# ------ Numeric Helpers ------ #

Number = int | float


def _finite(value: Number, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def _number(value: Number, name: str) -> float:
    return _finite(value, name)


def clamp(
    value: Number,
    minimum: Number,
    maximum: Number,
) -> float:
    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)

    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum")

    return max(minimum, min(value, maximum))


def lerp(
    start: Number,
    end: Number,
    amount: Number,
) -> float:
    return float(start) + (
        float(end) - float(start)
    ) * float(amount)


# ------ Point ------ #


@dataclass(frozen=True, slots=True)
class Point:
    """A two-dimensional point."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x",
            _finite(self.x, "x"),
        )
        object.__setattr__(
            self,
            "y",
            _finite(self.y, "y"),
        )

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x

        if index == 1:
            return self.y

        raise IndexError("Point index must be 0 or 1")

    def __add__(self, other: Point | Vector) -> Point:
        if isinstance(other, Point):
            return Point(
                self.x + other.x,
                self.y + other.y,
            )

        if isinstance(other, Vector):
            return Point(
                self.x + other.x,
                self.y + other.y,
            )

        return NotImplemented

    def __sub__(
        self,
        other: Point | Vector,
    ) -> Point | Vector:
        if isinstance(other, Point):
            return Vector(
                self.x - other.x,
                self.y - other.y,
            )

        if isinstance(other, Vector):
            return Point(
                self.x - other.x,
                self.y - other.y,
            )

        return NotImplemented

    def translated(
        self,
        dx: Number,
        dy: Number,
    ) -> Point:
        return Point(
            self.x + float(dx),
            self.y + float(dy),
        )

    def scaled(
        self,
        sx: Number,
        sy: Number | None = None,
        origin: Point | None = None,
    ) -> Point:
        if sy is None:
            sy = sx

        if origin is None:
            origin = Point(0, 0)

        return Point(
            origin.x + (self.x - origin.x) * float(sx),
            origin.y + (self.y - origin.y) * float(sy),
        )

    def distance_to(self, other: Point) -> float:
        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )

    def midpoint(self, other: Point) -> Point:
        return Point(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
        )

    def lerp(
        self,
        other: Point,
        amount: Number,
    ) -> Point:
        return Point(
            lerp(self.x, other.x, amount),
            lerp(self.y, other.y, amount),
        )

    def rounded(self, ndigits: int = 0) -> Point:
        return Point(
            round(self.x, ndigits),
            round(self.y, ndigits),
        )

    def floor(self) -> Point:
        return Point(
            math.floor(self.x),
            math.floor(self.y),
        )

    def ceil(self) -> Point:
        return Point(
            math.ceil(self.x),
            math.ceil(self.y),
        )

    def to_int(self) -> tuple[int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
        )

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


# ------ Vector ------ #


@dataclass(frozen=True, slots=True)
class Vector:
    """A two-dimensional mathematical vector."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x",
            _finite(self.x, "x"),
        )
        object.__setattr__(
            self,
            "y",
            _finite(self.y, "y"),
        )

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __add__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            return NotImplemented

        return Vector(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            return NotImplemented

        return Vector(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, scalar: Number) -> Vector:
        return Vector(
            self.x * float(scalar),
            self.y * float(scalar),
        )

    def __rmul__(self, scalar: Number) -> Vector:
        return self * scalar

    def __truediv__(self, scalar: Number) -> Vector:
        scalar = float(scalar)

        if scalar == 0:
            raise ZeroDivisionError(
                "Cannot divide a vector by zero"
            )

        return Vector(
            self.x / scalar,
            self.y / scalar,
        )

    def __neg__(self) -> Vector:
        return Vector(
            -self.x,
            -self.y,
        )

    @property
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    @property
    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def normalized(self) -> Vector:
        length = self.length

        if length == 0:
            return Vector(0, 0)

        return self / length

    def dot(self, other: Vector) -> float:
        return (
            self.x * other.x
            + self.y * other.y
        )

    def cross(self, other: Vector) -> float:
        return (
            self.x * other.y
            - self.y * other.x
        )

    def rotated(
        self,
        angle: Number,
    ) -> Vector:
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))

        return Vector(
            self.x * cosine - self.y * sine,
            self.x * sine + self.y * cosine,
        )

    def perpendicular(self) -> Vector:
        return Vector(
            -self.y,
            self.x,
        )

    def distance_to(self, other: Vector) -> float:
        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


# ------ Size ------ #


@dataclass(frozen=True, slots=True)
class Size:
    """A non-negative two-dimensional size."""

    width: int
    height: int

    def __post_init__(self) -> None:
        width = int(self.width)
        height = int(self.height)

        if width < 0 or height < 0:
            raise ValueError(
                "Size dimensions cannot be negative"
            )

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def __iter__(self) -> Iterator[int]:
        yield self.width
        yield self.height

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.width

        if index == 1:
            return self.height

        raise IndexError("Size index must be 0 or 1")

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0

        return self.width / self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    def scaled(
        self,
        scale: Number,
    ) -> Size:
        scale = float(scale)

        if scale < 0:
            raise ValueError("scale cannot be negative")

        return Size(
            round(self.width * scale),
            round(self.height * scale),
        )

    def fit_inside(
        self,
        maximum: Size,
        allow_upscale: bool = True,
    ) -> Size:
        if self.is_empty:
            return Size(0, 0)

        if maximum.is_empty:
            return Size(0, 0)

        scale = min(
            maximum.width / self.width,
            maximum.height / self.height,
        )

        if not allow_upscale:
            scale = min(scale, 1.0)

        return self.scaled(scale)

    def fill(
        self,
        minimum: Size,
        allow_upscale: bool = True,
    ) -> Size:
        if self.is_empty:
            return Size(0, 0)

        if minimum.is_empty:
            return Size(0, 0)

        scale = max(
            minimum.width / self.width,
            minimum.height / self.height,
        )

        if not allow_upscale:
            scale = min(scale, 1.0)

        return self.scaled(scale)

    def as_tuple(self) -> tuple[int, int]:
        return self.width, self.height


# ------ Rectangle ------ #


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (
            _finite(self.x, "x"),
            _finite(self.y, "y"),
            _finite(self.width, "width"),
            _finite(self.height, "height"),
        )

        if values[2] < 0 or values[3] < 0:
            raise ValueError(
                "Rectangle dimensions cannot be negative"
            )

        object.__setattr__(self, "x", values[0])
        object.__setattr__(self, "y", values[1])
        object.__setattr__(self, "width", values[2])
        object.__setattr__(self, "height", values[3])

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.width
        yield self.height

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
    def center(self) -> Point:
        return Point(
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def top_left(self) -> Point:
        return Point(self.left, self.top)

    @property
    def top_right(self) -> Point:
        return Point(self.right, self.top)

    @property
    def bottom_left(self) -> Point:
        return Point(self.left, self.bottom)

    @property
    def bottom_right(self) -> Point:
        return Point(self.right, self.bottom)

    @property
    def size(self) -> Size:
        return Size(
            max(0, math.ceil(self.width)),
            max(0, math.ceil(self.height)),
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def contains(
        self,
        point: Point | Number,
        y: Number | None = None,
    ) -> bool:
        if isinstance(point, Point):
            px = point.x
            py = point.y
        else:
            if y is None:
                raise TypeError(
                    "y is required when x is provided"
                )

            px = float(point)
            py = float(y)

        return (
            self.left <= px < self.right
            and self.top <= py < self.bottom
        )

    def contains_rect(self, other: Rect) -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def intersects(self, other: Rect) -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )

    def intersection(
        self,
        other: Rect,
    ) -> Rect | None:
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

    def union(self, other: Rect) -> Rect:
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)

        return Rect(
            left,
            top,
            right - left,
            bottom - top,
        )

    @classmethod
    def from_points(
        cls,
        first: Point,
        second: Point,
    ) -> Rect:
        left = min(first.x, second.x)
        top = min(first.y, second.y)
        right = max(first.x, second.x)
        bottom = max(first.y, second.y)

        return cls(
            left,
            top,
            right - left,
            bottom - top,
        )

    @classmethod
    def from_size(
        cls,
        size: Size,
        x: Number = 0,
        y: Number = 0,
    ) -> Rect:
        return cls(
            float(x),
            float(y),
            size.width,
            size.height,
        )

    @classmethod
    def empty(cls) -> Rect:
        return cls(0, 0, 0, 0)

    def translated(
        self,
        dx: Number,
        dy: Number,
    ) -> Rect:
        return Rect(
            self.x + float(dx),
            self.y + float(dy),
            self.width,
            self.height,
        )

    def inflated(
        self,
        horizontal: Number,
        vertical: Number | None = None,
    ) -> Rect:
        if vertical is None:
            vertical = horizontal

        horizontal = float(horizontal)
        vertical = float(vertical)

        return Rect(
            self.x - horizontal,
            self.y - vertical,
            self.width + horizontal * 2,
            self.height + vertical * 2,
        )

    def deflated(
        self,
        horizontal: Number,
        vertical: Number | None = None,
    ) -> Rect:
        return self.inflated(
            -float(horizontal),
            -float(horizontal if vertical is None else vertical),
        )

    def scaled(
        self,
        sx: Number,
        sy: Number | None = None,
        origin: Point | None = None,
    ) -> Rect:
        if sy is None:
            sy = sx

        if origin is None:
            origin = Point(0, 0)

        sx = float(sx)
        sy = float(sy)

        x1 = origin.x + (self.left - origin.x) * sx
        y1 = origin.y + (self.top - origin.y) * sy
        x2 = origin.x + (self.right - origin.x) * sx
        y2 = origin.y + (self.bottom - origin.y) * sy

        return Rect.from_points(
            Point(x1, y1),
            Point(x2, y2),
        )

    def clamp_point(self, point: Point) -> Point:
        return Point(
            clamp(point.x, self.left, self.right),
            clamp(point.y, self.top, self.bottom),
        )

    def distance_to(self, point: Point) -> float:
        closest = self.clamp_point(point)

        return closest.distance_to(point)

    def rounded(self, ndigits: int = 0) -> Rect:
        return Rect(
            round(self.x, ndigits),
            round(self.y, ndigits),
            round(self.width, ndigits),
            round(self.height, ndigits),
        )

    def floor(self) -> Rect:
        return Rect(
            math.floor(self.x),
            math.floor(self.y),
            math.floor(self.width),
            math.floor(self.height),
        )

    def ceil(self) -> Rect:
        return Rect(
            math.ceil(self.x),
            math.ceil(self.y),
            math.ceil(self.width),
            math.ceil(self.height),
        )

    def to_int(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.width)),
            int(round(self.height)),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )


# ------ Polygon Helpers ------ #


def bounding_box(
    points: Iterable[Point],
) -> Rect:
    points = tuple(points)

    if not points:
        return Rect.empty()

    left = min(point.x for point in points)
    top = min(point.y for point in points)
    right = max(point.x for point in points)
    bottom = max(point.y for point in points)

    return Rect(
        left,
        top,
        right - left,
        bottom - top,
    )


def polygon_area(
    points: Iterable[Point],
) -> float:
    points = tuple(points)

    if len(points) < 3:
        return 0.0

    total = 0.0

    for index, point in enumerate(points):
        next_point = points[
            (index + 1) % len(points)
        ]

        total += (
            point.x * next_point.y
            - next_point.x * point.y
        )

    return abs(total) / 2.0


def polygon_centroid(
    points: Iterable[Point],
) -> Point:
    points = tuple(points)

    if not points:
        raise ValueError(
            "Cannot calculate centroid of empty points"
        )

    area_factor = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for index, point in enumerate(points):
        next_point = points[
            (index + 1) % len(points)
        ]

        cross = (
            point.x * next_point.y
            - next_point.x * point.y
        )

        area_factor += cross

        centroid_x += (
            point.x + next_point.x
        ) * cross

        centroid_y += (
            point.y + next_point.y
        ) * cross

    if abs(area_factor) < 1e-12:
        return Point(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )

    factor = 1.0 / (3.0 * area_factor)

    return Point(
        centroid_x * factor,
        centroid_y * factor,
    )


def distance(
    first: Point,
    second: Point,
) -> float:
    return first.distance_to(second)


def midpoint(
    first: Point,
    second: Point,
) -> Point:
    return first.midpoint(second)


def rotate_point(
    point: Point,
    angle: Number,
    origin: Point | None = None,
) -> Point:
    if origin is None:
        origin = Point(0, 0)

    vector = point - origin

    if not isinstance(vector, Vector):
        raise TypeError("Invalid point subtraction")

    rotated = vector.rotated(angle)

    return origin + rotated


def transform_points(
    points: Iterable[Point],
    *,
    translation: Vector | None = None,
    scale: Number | tuple[Number, Number] = 1.0,
    rotation: Number = 0.0,
    origin: Point | None = None,
) -> list[Point]:
    if origin is None:
        origin = Point(0, 0)

    if isinstance(scale, tuple):
        sx, sy = scale
    else:
        sx = sy = scale

    result: list[Point] = []

    for point in points:
        transformed = point.scaled(
            sx,
            sy,
            origin,
        )

        transformed = rotate_point(
            transformed,
            rotation,
            origin,
        )

        if translation is not None:
            transformed = transformed + translation

        result.append(transformed)

    return result


# ------ Public API ------ #


__all__ = [
    "Number",
    "Point",
    "Rect",
    "Size",
    "Vector",
    "bounding_box",
    "clamp",
    "distance",
    "lerp",
    "midpoint",
    "polygon_area",
    "polygon_centroid",
    "rotate_point",
    "transform_points",
]
