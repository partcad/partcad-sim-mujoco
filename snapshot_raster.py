#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""Pictures of a simulated scene, and a video of the run, drawn on the CPU.

PartCAD asks every simulation for two pictures of the scene -- one before
anything moved and one when the run is over -- and says where to take them
from: ``request["snapshot"]``, read from the scene's own ``render: png:``
configuration (see 'wrappers/wrapper_simulate.py' in PartCAD itself). This is
what draws them. It also asks for a video of the whole run, from the same
viewpoint, at the frame rate and the resolution it names in
``request["video"]``; 'record' is what draws and encodes that.

**Why not the simulator's own renderer.** A simulator draws with OpenGL, and a
headless OpenGL needs EGL or OSMesa -- native libraries that no wheel carries
and that PartCAD's sandbox images do not have. A picture that can only be taken
on a machine with a GPU driver set up is a picture most runs never get. What is
drawn here is the simulator's *own* geometry at the simulator's *own* poses --
the meshes and primitives the engine compiled, placed where the engine left
them -- rasterized with numpy, so it needs nothing but numpy and works in every
sandbox the simulation itself works in.

**What the picture is.** An orthographic projection looking from the direction
``viewport_origin`` names towards the middle of what is drawn, with
``viewport_up`` up -- the same two numbers ``pc render`` reads, meaning the same
thing, so the snapshots of a scene are taken from where its pictures are. Both
pictures are framed alike, on everything either of them holds, which is what
makes the second readable against the first at a glance: a block that fell is
lower in the frame rather than re-centred. The floor is a checkerboard fixed to
the world for the same reason. Every frame of a video is framed alike too, on
everything any frame of it holds.

**How it is fast enough for a video.** A run of ten seconds at 25 frames a
second is 251 pictures, so a picture has to cost tens of milliseconds rather
than the better part of a second. Three things make it so: the triangles that
cover a few pixels each -- nearly all of a mesh's -- are filled together, many
at a time, rather than one Python call each ('Picture.add'); the floor,
which does not move, is drawn once for the whole video, and a frame finishes
again only the part of the picture something moved in ('Frames'); and the back
of a closed surface, which its front hides, is not drawn at all ('outward').

**The video.** WebM, with VP9 in it: the one pairing every browser plays,
including the one a Visual Studio Code webview is, which plays no H.264. It is
encoded by FFmpeg, as shipped in the ``imageio-ffmpeg`` wheel -- a program with
no Python of its own and a wheel for every platform a sandbox runs on, which is
why that is the one requirement a video adds to a plugin.

This file is the same in every PartCAD simulation plugin -- the MuJoCo one and
the Gazebo one -- and is kept identical rather than shared: a plugin is a
package of its own and does not reach into another. What differs between them
is only where the triangles come from.
"""

import os
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# What is behind everything, the two squares of the floor, and what a body is
# drawn in when the scene says nothing about its colour.
BACKGROUND = (0.97, 0.97, 0.98)
GROUND_DARKEN = 0.88
DEFAULT_GROUND = (0.84, 0.85, 0.87)
DEFAULT_COLOR = (0.62, 0.66, 0.72)

# Light from over the viewer's shoulder, plus enough ambient light that a face
# turned away is dark rather than black. They add up to more than one so that a
# face turned to the light shows a colour a little brighter than it was stated
# in, the way a lit surface does: MuJoCo's default grey is 0.5, and drawn at no
# more than its own value it reads as a shadow.
AMBIENT = 0.45
DIFFUSE = 0.75

# How much of the picture is left around what is framed, on each side.
MARGIN = 0.12

# Each picture is drawn this many times larger and averaged down, which is what
# keeps an edge from being a staircase.
SUPERSAMPLE = 2

# How deep a square of the floor is on the picture at the least, in pixels.
MIN_SQUARE_PIXELS = 10.0

# How dark an outline is, as a fraction of the colour it is drawn over.
OUTLINE = 0.3

# Pixel ids: what is at a pixel, for drawing outlines between things.
_BACKGROUND_ID = -1
_GROUND_ID = 0

# The largest triangle, in pixels of its bounding box, that is filled together
# with others rather than on its own. Most of a mesh is far smaller, and filling
# one costs a Python call however few pixels it covers; a larger one -- a floor
# square, a face of a block filling the picture -- is few enough, and its pixels
# many enough, to be worth a call.
BATCH_AREA = 64 * 64

# The widths and heights, in pixels, the bounding boxes of a batch are rounded
# up to: powers of two and the halfway steps between them, so that a box is
# tested over no more than half as much again as it holds on each side.
BATCH_SIZES = np.unique(np.concatenate([2 ** np.arange(21), 3 * 2 ** np.arange(20)]))

# How many pixels one batch of triangles tests at once, at most: what bounds
# the memory filling them takes.
BATCH_PIXELS = 1 << 18

# The videos this draws, by the 'format' PartCAD asks for: the encoder, and what
# it is told. VP9 at a constant quality, which for flat shading and a still
# background is a few hundred kilobytes for ten seconds; 'good' at speed 5 is a
# second or two of encoding for that, a small part of drawing the frames. A
# keyframe every two seconds, so that a player seeking back and forth has never
# far to decode from.
VIDEO_FORMATS = {
    "webm": {
        "codec": "libvpx-vp9",
        "parameters": [
            "-deadline",
            "good",
            "-cpu-used",
            "5",
            "-row-mt",
            "1",
            "-b:v",
            "0",
            "-crf",
            "31",
            "-g",
            "50",
            "-f",
            "webm",
        ],
    },
}

# What a video is when the request does not say: PartCAD always does.
DEFAULT_FPS = 25.0
DEFAULT_VIDEO_SIZE = (1280, 720)

# How many frames are drawn before the time the rest will take is estimated,
# and a video that would not be finished within the run's budget is given up.
ESTIMATE_AFTER = 5

# The largest video whose frames are drawn supersampled, in pixels: full HD.
VIDEO_SUPERSAMPLED_PIXELS = 1920 * 1080

# How many frames of a video are drawn at once, at most, on as many threads.
# numpy lets go of the interpreter while it works through an array, so a
# second thread draws nearly a second frame in the time; past four they wait
# for one another more than they draw.
VIDEO_THREADS = 4

# What the pictures being drawn at once may take between them, in bytes, and
# what a picture takes for each pixel it is drawn in: a colour, a depth and an
# id.
VIDEO_MEMORY = 512 * 1024 * 1024
PICTURE_BYTES_PER_PIXEL = 3 * 4 + 4 + 4


class OutOfTime(Exception):
    """The video would not be finished within the time the run has left."""


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("a direction cannot be the zero vector")
    return vector / length


class Camera:
    """An orthographic camera looking from a direction, framed on some points."""

    def __init__(self, viewport_origin, viewport_up, width, height):
        toward = _unit(viewport_origin)
        up = _unit(viewport_up)
        forward = -toward
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-9:
            # 'up' along the line of sight names no orientation; any
            # perpendicular will do, and PartCAD refuses such a pair anyway.
            right = np.cross(forward, (1.0, 0.0, 0.0) if abs(forward[0]) < 0.9 else (0.0, 1.0, 0.0))
        right = _unit(right)
        # Rows: screen x (right), screen y (up), and depth towards the viewer.
        self.axes = np.stack([right, np.cross(right, forward), toward])
        self.width = int(width)
        self.height = int(height)
        self.center = np.zeros(2)
        self.scale = 1.0

    def project(self, points):
        """Points as (screen x, screen y, depth), depth growing towards the viewer."""
        return np.asarray(points, dtype=float) @ self.axes.T

    def bounds(self, points):
        """Where these points reach on the picture, as the lowest and highest (x, y), or None for none."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return None
        projected = self.project(points)[:, :2]
        return projected.min(axis=0), projected.max(axis=0)

    def frame(self, points):
        """Fit these points into the picture, with a margin around them."""
        self.fit(self.bounds(points))

    def fit(self, bounds):
        """Fit what 'bounds' reaches into the picture, with a margin around it."""
        if bounds is None:
            bounds = self.bounds(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        low, high = bounds
        size = np.maximum(high - low, 1e-9)
        usable = 1.0 - 2.0 * MARGIN
        self.scale = usable * min(self.width / size[0], self.height / size[1])
        self.center = (low + high) / 2.0

    def extent(self):
        """How far the picture reaches from its centre, in scene units."""
        return 0.5 * np.hypot(self.width, self.height) / self.scale

    def to_pixels(self, projected, factor=1):
        """Projected points as pixel coordinates (x right, y down) in a picture 'factor' times larger."""
        x = (projected[..., 0] - self.center[0]) * self.scale * factor + self.width * factor / 2.0
        y = self.height * factor / 2.0 - (projected[..., 1] - self.center[1]) * self.scale * factor
        return x, y


def _union(first, second):
    """The bounds reaching over both, either of which may be None."""
    if first is None:
        return second
    if second is None:
        return first
    return np.minimum(first[0], second[0]), np.maximum(first[1], second[1])


class Picture:
    """A depth-buffered canvas, drawn into one triangle at a time."""

    def __init__(self, camera, supersample=SUPERSAMPLE):
        self.camera = camera
        self.supersample = int(supersample)
        self.width = camera.width * self.supersample
        self.height = camera.height * self.supersample
        self.color = np.empty((self.height, self.width, 3), dtype=np.float32)
        self.color[:] = BACKGROUND
        # Single precision: a depth is told from another to a millionth of
        # what the picture shows, far finer than anything drawn in it, and the
        # frames of a video copy it every time.
        self.depth = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        self.ids = np.full((self.height, self.width), _BACKGROUND_ID, dtype=np.int32)
        axes = camera.axes
        self.light = _unit(axes[2] + 0.6 * axes[1] + 0.3 * axes[0])
        # What has been drawn into since it was last cleared, as (top, bottom,
        # left, right), inclusive; None for nothing. What the frames of a video
        # put back and finish again, rather than the whole of every picture.
        self.dirty = None

    def copy(self):
        """Another picture holding what this one does, to draw more into."""
        other = Picture.__new__(Picture)
        other.__dict__.update(self.__dict__)
        other.color = self.color.copy()
        other.depth = self.depth.copy()
        other.ids = self.ids.copy()
        return other

    def restore(self, other, region):
        """Put back what 'other' holds over 'region' (top, bottom, left, right), and forget what was drawn."""
        top, bottom, left, right = region
        rows, columns = slice(top, bottom + 1), slice(left, right + 1)
        self.color[rows, columns] = other.color[rows, columns]
        self.depth[rows, columns] = other.depth[rows, columns]
        self.ids[rows, columns] = other.ids[rows, columns]
        self.dirty = None

    def shade(self, triangles, color):
        """One colour per triangle, lit by how squarely it faces the light."""
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        # Both sides: nothing says which way a face of an arbitrary mesh points.
        facing = np.abs((normals / lengths[:, None]) @ self.light)
        return np.clip(np.asarray(color, dtype=float)[None, :] * (AMBIENT + DIFFUSE * facing)[:, None], 0.0, 1.0)

    def add(self, triangles, colors, ident, bias=0.0):
        """Draw triangles (N, 3, 3) in the colours (N, 3), as the thing 'ident' names.

        Every pixel of a triangle's bounding box is tested, and what is inside
        and nearer than what the pixel holds already is kept. The boxes are
        tested many at a time rather than a triangle at a time: one Python call
        per triangle is what a mesh of thousands of them, each covering a few
        pixels, would otherwise cost. A box larger than 'BATCH_AREA' pixels has
        pixels enough to be worth a call of its own, and gets one.
        """
        triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        if len(triangles) == 0:
            return
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        projected = self.camera.project(triangles)
        xs, ys = self.camera.to_pixels(projected, self.supersample)
        zs = projected[..., 2] - bias
        area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (xs[:, 2] - xs[:, 0]) * (ys[:, 1] - ys[:, 0])
        finite = np.isfinite(xs).all(axis=1) & np.isfinite(ys).all(axis=1) & np.isfinite(zs).all(axis=1)
        with np.errstate(invalid="ignore"):
            # Clipped to the picture before they are made whole numbers, so
            # that a vertex far off it is not a number too large to be one.
            left = np.clip(np.floor(xs.min(axis=1)), 0, self.width)
            right = np.clip(np.ceil(xs.max(axis=1)), -1, self.width - 1)
            top = np.clip(np.floor(ys.min(axis=1)), 0, self.height)
            bottom = np.clip(np.ceil(ys.max(axis=1)), -1, self.height - 1)
        drawn = np.nonzero(finite & (np.abs(area) >= 1e-12) & (left <= right) & (top <= bottom))[0]
        if not len(drawn):
            return
        left, right, top, bottom = (v[drawn].astype(np.int64) for v in (left, right, top, bottom))
        self.dirty = _box_union(self.dirty, (int(top.min()), int(bottom.max()), int(left.min()), int(right.max())))
        # Each weight and the depth as a linear function of where a pixel is in
        # its triangle's box: see '_linear'.
        forms = _linear(xs[drawn], ys[drawn], zs[drawn], area[drawn], left, top)
        colors = colors[drawn]

        # The smallest rectangle of 'BATCH_SIZES' that holds each box: a batch
        # is the boxes one rectangle holds, so that it tests about as many
        # pixels as its boxes hold -- a long thin triangle's included.
        wide = np.searchsorted(BATCH_SIZES, right - left + 1)
        high = np.searchsorted(BATCH_SIZES, bottom - top + 1)
        last = len(BATCH_SIZES) - 1
        large = BATCH_SIZES[np.minimum(wide, last)] * BATCH_SIZES[np.minimum(high, last)] > BATCH_AREA
        for index in np.nonzero(large)[0]:
            self._triangle(forms[index], colors[index], ident, (left[index], right[index], top[index], bottom[index]))
        small = np.nonzero(~large)[0]
        if not len(small):
            return
        group = wide[small] * len(BATCH_SIZES) + high[small]
        order = small[np.argsort(group, kind="stable")]
        group = np.sort(group, kind="stable")
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for first, end in zip(starts, ends):
            members = order[first:end]
            columns = int(BATCH_SIZES[wide[members[0]]])
            rows = int(BATCH_SIZES[high[members[0]]])
            step = max(1, BATCH_PIXELS // (columns * rows))
            for start in range(0, len(members), step):
                part = members[start : start + step]
                self._fill(
                    forms[part], colors[part], (left[part], right[part], top[part], bottom[part]), columns, rows, ident
                )

    def _triangle(self, form, color, ident, box):
        """Fill one triangle, testing every pixel of its bounding box (left, right, top, bottom).

        Row by row: on each row of the box, each of the three weights is
        nought or more on one side of a point, so the triangle covers the span
        between the three -- worked out for every row at once, and far cheaper
        than working out each weight at every pixel.
        """
        left, right, top, bottom = (int(v) for v in box)
        u = np.arange(right - left + 1) + 0.5
        v = np.arange(bottom - top + 1) + 0.5
        start = np.full(len(v), -np.inf)
        end = np.full(len(v), np.inf)
        for a, b, c in (form[0:3], form[3:6], form[6:9]):
            rest = b * v + c
            if a > 0:
                start = np.maximum(start, -rest / a)
            elif a < 0:
                end = np.minimum(end, -rest / a)
            else:
                end = np.where(rest >= 0, end, -np.inf)
        inside = (u >= start[:, None]) & (u <= end[:, None])
        depth = (form[9] * u + (form[10] * v[:, None] + form[11])).astype(self.depth.dtype)
        rows, columns = slice(top, bottom + 1), slice(left, right + 1)
        region = self.depth[rows, columns]
        nearer = inside & (depth > region)
        np.copyto(region, depth, where=nearer)
        np.copyto(self.color[rows, columns], np.asarray(color, dtype=np.float32), where=nearer[..., None])
        np.copyto(self.ids[rows, columns], ident, where=nearer)

    def _fill(self, forms, colors, boxes, columns, rows, ident):
        """Fill a batch of triangles whose bounding boxes are each at most 'columns' by 'rows' pixels.

        'boxes' is (left, right, top, bottom), an array each. Every pixel
        inside a triangle is a fragment, and of the fragments that land on one
        pixel the nearest is kept, if it is nearer than what the pixel holds
        already: the depth test '_triangle' makes, for all of them at once.
        """
        left, right, top, bottom = boxes
        u = (np.arange(columns) + 0.5)[None, None, :]
        v = (np.arange(rows) + 0.5)[None, :, None]
        f = forms[:, :, None, None]
        # One array the size of the batch per weight: each term on its own is
        # a row or a column.
        w0 = f[:, 0] * u + (f[:, 1] * v + f[:, 2])
        inside = w0 >= 0
        w1 = f[:, 3] * u + (f[:, 4] * v + f[:, 5])
        inside &= w1 >= 0
        w2 = f[:, 6] * u + (f[:, 7] * v + f[:, 8])
        inside &= w2 >= 0
        inside &= (u < (right - left + 1)[:, None, None]) & (v < (bottom - top + 1)[:, None, None])
        hit, row, column = np.nonzero(inside)
        if not len(hit):
            return
        chosen = forms[hit]
        depth = (chosen[:, 9] * (column + 0.5) + chosen[:, 10] * (row + 0.5) + chosen[:, 11]).astype(self.depth.dtype)
        pixel = (top[hit] + row) * self.width + left[hit] + column
        flat = self.depth.reshape(-1)
        before = flat[pixel]
        np.maximum.at(flat, pixel, depth)
        # The nearest fragment of each pixel is the one whose depth the pixel
        # now holds; a fragment no nearer than what was there already is not.
        kept = (depth == flat[pixel]) & (depth > before)
        self.color.reshape(-1, 3)[pixel[kept]] = colors[hit[kept]]
        self.ids.reshape(-1)[pixel[kept]] = ident

    def whole(self, region):
        """'region' (top, bottom, left, right), widened by the outline around it and out to whole pixels.

        Whole pixels of the picture asked for, that is: a square of
        'supersample' pixels on a side of the picture drawn.
        """
        factor = self.supersample
        top, bottom, left, right = region
        top = max(top - 1, 0) // factor * factor
        left = max(left - 1, 0) // factor * factor
        bottom = min((min(bottom + 1, self.height - 1) // factor + 1) * factor, self.height) - 1
        right = min((min(right + 1, self.width - 1) // factor + 1) * factor, self.width) - 1
        return top, bottom, left, right

    def finish(self, region=None):
        """The picture at the size asked for, as (height, width, 3) bytes.

        Or only the part of it 'region' covers, in the pixels it is drawn in,
        once it is made 'whole' -- which is where in the picture asked for the
        part returned goes, each of its numbers divided by 'supersample'.
        """
        top, bottom, left, right = (0, self.height - 1, 0, self.width - 1) if region is None else self.whole(region)
        # The ids a pixel further each way, for the outlines along the edges.
        outer_top, outer_left = max(top - 1, 0), max(left - 1, 0)
        ids = self.ids[outer_top : min(bottom + 2, self.height), outer_left : min(right + 2, self.width)]
        edge = np.zeros(ids.shape, dtype=bool)
        # An outline wherever two things meet, unless both are the floor or
        # the background: the floor's squares are not things.
        across = (ids[:, 1:] != ids[:, :-1]) & (np.maximum(ids[:, 1:], ids[:, :-1]) > _GROUND_ID)
        edge[:, 1:] |= across
        edge[:, :-1] |= across
        down = (ids[1:, :] != ids[:-1, :]) & (np.maximum(ids[1:, :], ids[:-1, :]) > _GROUND_ID)
        edge[1:, :] |= down
        edge[:-1, :] |= down
        edge = edge[top - outer_top : bottom + 1 - outer_top, left - outer_left : right + 1 - outer_left]
        color = (
            self.color[top : bottom + 1, left : right + 1]
            * np.where(edge, np.float32(OUTLINE), np.float32(1.0))[..., None]
        )
        factor = self.supersample
        if factor > 1:
            # Each pixel the mean of the square of pixels it was drawn as.
            total = None
            for dy in range(factor):
                for dx in range(factor):
                    part = color[dy::factor, dx::factor]
                    total = part.copy() if total is None else np.add(total, part, out=total)
            color = total * np.float32(1.0 / (factor * factor))
        color *= 255.0
        color += 0.5
        np.clip(color, 0.0, 255.0, out=color)
        return color.astype(np.uint8)


def _linear(xs, ys, zs, area, left, top):
    """Each triangle's barycentric weights and depth, as linear functions of a pixel's place in its box.

    (N, 12): for each of w0, w1, w2 and the depth, the ``a``, ``b`` and ``c`` of
    ``a * u + b * v + c``, where (u, v) is the centre of a pixel counted from
    the corner (left, top) of the triangle's box. Counted from there rather
    than from the corner of the picture, the numbers stay small, and so does
    what rounding them loses. w0 is the signed area opposite the first vertex,
    over the triangle's, and so on.
    """
    x = xs - left[:, None]
    y = ys - top[:, None]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
    a0, b0, c0 = (y1 - y2) / area, (x2 - x1) / area, (x1 * y2 - x2 * y1) / area
    a1, b1, c1 = (y2 - y0) / area, (x0 - x2) / area, (x2 * y0 - x0 * y2) / area
    a2, b2, c2 = -(a0 + a1), -(b0 + b1), 1.0 - c0 - c1
    z0, z1, z2 = zs[:, 0], zs[:, 1], zs[:, 2]
    depth = [a0 * z0 + a1 * z1 + a2 * z2, b0 * z0 + b1 * z1 + b2 * z2, c0 * z0 + c1 * z1 + c2 * z2]
    return np.stack([a0, b0, c0, a1, b1, c1, a2, b2, c2] + depth, axis=1)


def _box_union(first, second):
    """The (top, bottom, left, right) box over both, either of which may be None."""
    if first is None:
        return second
    if second is None:
        return first
    return (min(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), max(first[3], second[3]))


#
# Shapes
#


def box(half):
    """A box of these half-extents, centred on the origin, as triangles."""
    hx, hy, hz = (float(v) for v in half[:3])
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [
        (0, 1, 3, 2),  # -x
        (4, 6, 7, 5),  # +x
        (0, 4, 5, 1),  # -y
        (2, 3, 7, 6),  # +y
        (0, 2, 6, 4),  # -z
        (1, 5, 7, 3),  # +z
    ]
    return np.array([corners[[a, b, c]] for a, b, c, d in faces] + [corners[[a, c, d]] for a, b, c, d in faces])


def ellipsoid(rx, ry, rz, segments=24, rings=12):
    """An ellipsoid of these radii, centred on the origin, as triangles."""
    return _quads(_globe(rx, ry, rz, np.linspace(0.0, np.pi, rings + 1), segments))


def cylinder(radius, half_length, segments=24):
    """A capped cylinder along z, centred on the origin, as triangles."""
    ring = _ring(radius, segments)
    side = _quads(np.stack([ring + (0, 0, -half_length), ring + (0, 0, half_length)]))
    caps = []
    for z, turn in ((-half_length, 1), (half_length, -1)):
        # Each cap wound the way the side is, so that the three are one closed
        # surface (see 'outward').
        centre = np.array([0.0, 0.0, z])
        for i in range(segments):
            caps.append([centre, ring[i] + (0, 0, z), ring[i + 1] + (0, 0, z)][::turn])
    return np.concatenate([side, np.array(caps)])


def capsule(radius, half_length, segments=24, rings=12):
    """A cylinder with hemispherical ends, along z, centred on the origin."""
    # A sphere pulled apart at its equator: the rows of the upper half up and
    # those of the lower half down, the equator in both, so that the band
    # between its two copies is the cylinder.
    half = max(1, (rings + 1) // 2)
    theta = np.linspace(0.0, np.pi, 2 * half + 1)
    grid = _globe(radius, radius, radius, np.concatenate([theta[: half + 1], theta[half:]]), segments)
    grid[: half + 1, :, 2] += half_length
    grid[half + 1 :, :, 2] -= half_length
    return _quads(grid)


def _ring(radius, segments):
    """A circle in the xy plane as 'segments' + 1 points, the last of them exactly the first."""
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    ring = np.stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)], axis=-1)
    ring[-1] = ring[0]
    return ring


def _globe(rx, ry, rz, theta, segments):
    """The points of an ellipsoid at these polar angles, as a (rows, segments + 1, 3) grid.

    Welded where it closes -- the last column exactly the first, and every
    point of a pole exactly the pole -- rather than as near as a sine of pi
    gets to nought, so that the surface is closed (see 'outward').
    """
    sine = np.sin(theta)
    sine[np.isclose(theta, 0.0) | np.isclose(theta, np.pi)] = 0.0
    ring = _ring(1.0, segments)
    return np.stack(
        [
            rx * sine[:, None] * ring[None, :, 0],
            ry * sine[:, None] * ring[None, :, 1],
            rz * np.cos(theta)[:, None] * np.ones(len(ring))[None, :],
        ],
        axis=-1,
    )


def _quads(grid):
    """Two triangles per cell of a (rows, columns, 3) grid of points."""
    a = grid[:-1, :-1].reshape(-1, 3)
    b = grid[1:, :-1].reshape(-1, 3)
    c = grid[1:, 1:].reshape(-1, 3)
    d = grid[:-1, 1:].reshape(-1, 3)
    return np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])


def place(triangles, rotation, translation, scale=1.0):
    """Triangles in a local frame, moved into the world: ``R (s p) + t``."""
    return (np.asarray(triangles, dtype=float) * scale) @ np.asarray(rotation, dtype=float).T + np.asarray(
        translation, dtype=float
    )


def outward(triangles):
    """'triangles' turned to face out of what they enclose, and whether they enclose anything.

    A surface that is closed -- every edge shared by two triangles, which go
    along it in opposite directions -- has an inside and an outside, and the
    triangles facing away from whoever looks at it are hidden by the ones
    facing them, from wherever it is looked at. So they need not be drawn,
    which is half of what such a surface costs. Which side is out is what the
    volume the triangles enclose says: positive for triangles wound
    anticlockwise as seen from outside. A surface that is not closed, or is not
    wound one way throughout, is returned as it is and every triangle of it is
    drawn: nothing says which of its faces are hidden. Triangles of no area are
    left out either way, since they draw nothing.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangles = triangles[np.linalg.norm(normals, axis=1) > 0]
    if len(triangles) < 4:
        return triangles, False
    points, index = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    # Corners that were apart and are one point once rounded are a triangle of
    # no area too.
    proper = (index[:, 0] != index[:, 1]) & (index[:, 1] != index[:, 2]) & (index[:, 2] != index[:, 0])
    triangles, index = triangles[proper], index[proper]
    if len(triangles) < 4:
        return triangles, False
    count = len(points)
    edges = np.concatenate([index[:, [0, 1]], index[:, [1, 2]], index[:, [2, 0]]])
    forward = np.sort(edges[:, 0] * count + edges[:, 1])
    backward = np.sort(edges[:, 1] * count + edges[:, 0])
    if (forward[1:] == forward[:-1]).any() or not np.array_equal(forward, backward):
        return triangles, False
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
    if not volume:
        return triangles, False
    return (triangles if volume > 0 else triangles[:, ::-1]), True


def read_stl(path):
    """The triangles of an STL file, binary or ASCII, as (N, 3, 3)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * count == len(data):
            records = np.frombuffer(data, dtype=np.uint8, count=50 * count, offset=84).reshape(count, 50)
            return records[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(float)
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in data.decode("utf-8", "replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    return np.array(vertices, dtype=float).reshape(-1, 3, 3)


#
# The pictures
#


def _ground(picture, plane, ident):
    """A checkerboard on a plane, out to the edges of the picture.

    The squares are fixed to the plane, not to the picture, so that what moved
    can be read against them. Their size is a round number near a tenth of what
    the picture shows, so a scene of millimetres and one of metres both get a
    floor that reads as one.
    """
    rotation = np.asarray(plane["axes"], dtype=float)
    origin = np.asarray(plane["origin"], dtype=float)
    normal = rotation[:, 2]
    if abs(normal @ picture.camera.axes[2]) < 1e-6:
        # Seen edge on: a plane that is a line, and nothing to draw.
        return
    camera = picture.camera
    reach = camera.extent()
    # How squarely the floor faces the viewer: one looking straight down, near
    # nothing at a grazing angle, where a square is drawn that much flatter.
    facing = abs(normal @ camera.axes[2])
    # Centred under the middle of the picture, so it reaches the edges: where
    # the line of sight through the middle meets the plane. The middle of the
    # picture as the camera states it is a point on the plane through the
    # origin that faces the camera, which for a scene far from the origin is
    # nowhere near the floor under it. Seen edge on is excluded above.
    toward = camera.axes[2]
    middle = camera.center[0] * camera.axes[0] + camera.center[1] * camera.axes[1]
    middle = middle + toward * (((origin - middle) @ normal) / (toward @ normal))
    local = rotation.T @ (middle - origin)
    # A round number (1, 2 or 5 times a power of ten) near a tenth of what the
    # picture shows, and large enough that a square seen this flat is still a
    # few pixels deep rather than a stripe of aliasing.
    smallest = max(reach / 5.0, MIN_SQUARE_PIXELS / (camera.scale * max(facing, 1e-3)))
    tile = 10.0 ** np.floor(np.log10(smallest))
    for step in (2.0, 5.0, 10.0):
        if tile >= smallest:
            break
        tile = 10.0 ** np.floor(np.log10(smallest)) * step
    span = 3.0 * reach / max(facing, 0.2)
    count = int(min(np.ceil(span / tile), 60))
    first_x = np.floor(local[0] / tile) - count
    first_y = np.floor(local[1] / tile) - count
    color = np.asarray(plane.get("color") or DEFAULT_GROUND, dtype=float)
    triangles, colors = [], []
    for i in range(2 * count):
        for j in range(2 * count):
            x0, y0 = (first_x + i) * tile, (first_y + j) * tile
            corners = np.array([[x0, y0, 0.0], [x0 + tile, y0, 0.0], [x0 + tile, y0 + tile, 0.0], [x0, y0 + tile, 0.0]])
            square = place(np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]]), rotation, origin)
            shade = color if (int(first_x) + int(first_y) + i + j) % 2 == 0 else color * GROUND_DARKEN
            triangles.append(square)
            colors.extend([shade, shade])
    # Pushed back a hair, so a face resting on the floor is drawn over it.
    picture.add(np.concatenate(triangles), np.array(colors), ident, bias=1e-4 * reach)


def _planes_key(planes):
    """What a scene's planes are, as something a drawn floor can be looked up by."""
    return tuple(
        (
            np.asarray(plane["origin"], dtype=float).tobytes(),
            np.asarray(plane["axes"], dtype=float).tobytes(),
            None if plane.get("color") is None else np.asarray(plane["color"], dtype=float).tobytes(),
        )
        for plane in planes
    )


def _floor(camera, planes, supersample=SUPERSAMPLE):
    """A picture with nothing in it but the floor."""
    picture = Picture(camera, supersample)
    for plane in planes:
        _ground(picture, plane, _GROUND_ID)
    return picture


def _solids(picture, scene):
    """Draw a scene's solids into a picture.

    A solid that says it is ``closed`` -- its triangles turned outward by
    'outward' -- is drawn without the triangles facing away from the camera.
    """
    toward = picture.camera.axes[2]
    for solid in scene.get("solids") or []:
        triangles = np.asarray(solid["triangles"], dtype=float).reshape(-1, 3, 3)
        if solid.get("closed"):
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            triangles = triangles[normals @ toward > 0]
        color = solid.get("color")
        color = DEFAULT_COLOR if color is None else color
        picture.add(triangles, picture.shade(triangles, color), _GROUND_ID + 1 + int(solid.get("id", 0)))


def draw(camera, scene, supersample=SUPERSAMPLE):
    """One picture of a scene: ``{"solids": [...], "planes": [...]}``.

    A solid is ``{"triangles": (N, 3, 3), "color": (r, g, b), "id": int}``, in
    world coordinates, and ``"closed": True`` where 'outward' said its triangles
    are a closed surface and turned them outward; solids sharing an id are one
    thing, and get no outline between them. A plane is ``{"origin": (3,), "axes": (3, 3), "color": ...}``,
    its normal the third column of 'axes'.
    """
    picture = _floor(camera, scene.get("planes") or [], supersample)
    _solids(picture, scene)
    return picture.finish()


class Frames:
    """The pictures of a video, one after another, drawn with one camera.

    The floor is drawn once and kept, finished, as the picture every frame
    starts from; each frame then draws its solids over a copy of it, and
    finishes again only what it drew over and what the frame before it did --
    which is what was moved from and to. The rest of the picture is the floor's
    as it was finished, and copying it costs next to nothing.

    One of these draws one frame at a time. Several, drawing alternate frames
    at once, share their floors through ``floors`` -- a dict and the lock that
    guards it -- so that it is drawn once for all of them.
    """

    def __init__(self, camera, supersample=SUPERSAMPLE, floors=None):
        self.camera = camera
        self.supersample = supersample
        self.floors = floors if floors is not None else ({}, threading.Lock())
        self.key = None

    def _floor(self, planes):
        """The picture of nothing but these planes, and that picture finished.

        Drawn once for every 'Frames' sharing them, whichever asks first.
        """
        kept, lock = self.floors
        key = _planes_key(planes)
        with lock:
            if key not in kept:
                if len(kept) > 1:
                    # A floor that moves is a floor drawn every frame; there is
                    # no point keeping every one of them.
                    kept.clear()
                floor = _floor(self.camera, planes, self.supersample)
                kept[key] = (floor, floor.finish())
            return key, kept[key]

    def draw(self, scene):
        """The next frame: a picture of 'scene', as (height, width, 3) bytes."""
        key, (floor, empty) = self._floor(scene.get("planes") or [])
        if key != self.key:
            # The first frame, or a floor that moved: drawn on afresh.
            self.key = key
            self.floor = floor
            self.empty = empty
            self.work = floor.copy()
            self.previous = None
        work = self.work
        if self.previous is not None:
            work.restore(self.floor, self.previous)
        _solids(work, scene)
        image = self.empty.copy()
        region = _box_union(self.previous, work.dirty)
        if region is not None:
            top, bottom, left, right = (value // work.supersample for value in work.whole(region))
            image[top : bottom + 1, left : right + 1] = work.finish(region)
        self.previous = work.dirty
        return image


def write_png(path, image):
    """An (height, width, 3) array of bytes, written as an RGB PNG."""
    height, width = image.shape[:2]
    rows = np.ascontiguousarray(image, dtype=np.uint8).reshape(height, width * 3)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def _solid_points(scene):
    """Every vertex of a scene's solids, as (N, 3)."""
    points = [np.asarray(solid["triangles"], dtype=float).reshape(-1, 3) for solid in scene.get("solids") or []]
    return np.concatenate(points) if points else np.zeros((0, 3))


def take(path, snapshot, scenes):
    """Draw each scene of ``scenes`` ({moment: scene}) as ``snapshot`` asks.

    ``snapshot`` is the request PartCAD sends (see the module docstring); what
    comes back is what goes in the result as ``snapshots``: each moment drawn,
    and the file it was written to, relative to ``path``. All of them are framed
    alike -- on every solid any of them holds -- so they can be read against
    each other.
    """
    if (snapshot.get("format") or "png").lower() != "png":
        raise ValueError("only PNG snapshots can be drawn here, not '%s'" % snapshot.get("format"))
    files = snapshot.get("files") or {}
    camera = Camera(
        snapshot.get("viewport_origin") or (100.0, -100.0, 100.0),
        snapshot.get("viewport_up") or (0.0, 0.0, 1.0),
        int(snapshot.get("width") or 512),
        int(snapshot.get("height") or 512),
    )
    bounds = None
    for moment, scene in scenes.items():
        if moment in files:
            bounds = _union(bounds, camera.bounds(_solid_points(scene)))
    camera.fit(bounds)

    written = {}
    for moment, scene in scenes.items():
        name = files.get(moment)
        if not name:
            continue
        write_png(os.path.join(path, name), draw(camera, scene))
        written[moment] = name
    return written


#
# The video
#


def frame_times(duration, fps):
    """The instants a video of a run of 'duration' seconds shows, at 'fps' frames a second.

    The first is the start of the run, and the last is no later than its end:
    a frame for each 1/fps seconds the run lasted, which is what makes the video
    play the run in real time.
    """
    duration = float(duration)
    fps = float(fps)
    if not (duration >= 0 and fps > 0):
        return []
    # A hair under the next whole frame is that frame: 10 s at 25 fps is 251
    # frames whatever 10 * 25 rounds to.
    count = int(np.floor(duration * fps + 1e-6)) + 1
    return [index / fps for index in range(count)]


def _even(value, default):
    """A size of a video, in pixels: a positive, even number, which is what its chroma is halved by."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    if value <= 0:
        value = default
    return max(2, value - value % 2)


def record(path, video, count, scene_at, deadline=None):
    """Draw 'count' frames, ``scene_at(index)`` each, into the video ``video`` asks for.

    ``video`` is the request PartCAD sends::

        {"format": "webm", "file": "video.webm", "fps": 25, "width": 1280, "height": 720,
         "viewport_origin": [100, -100, 100], "viewport_up": [0, 0, 1]}

    Every frame is framed alike, on every solid any frame holds, and the floor
    is drawn once for all of them. ``deadline`` is when the run's budget is
    spent (``time.monotonic()``): a video that would not be finished by then is
    given up rather than allowed to cost the run its answer, and 'OutOfTime'
    says so. Returns the file written, relative to ``path``.
    """
    started = time.monotonic()
    kind = (video.get("format") or "webm").lower()
    if kind not in VIDEO_FORMATS:
        raise ValueError(
            "only %s videos can be recorded here, not '%s'" % (", ".join(sorted(VIDEO_FORMATS)), video.get("format"))
        )
    if count < 1:
        raise ValueError("a video needs a frame, and the run gave none")
    name = video.get("file") or "video.%s" % kind
    fps = float(video.get("fps") or DEFAULT_FPS)
    width = _even(video.get("width"), DEFAULT_VIDEO_SIZE[0])
    height = _even(video.get("height"), DEFAULT_VIDEO_SIZE[1])
    camera = Camera(
        video.get("viewport_origin") or (100.0, -100.0, 100.0),
        video.get("viewport_up") or (0.0, 0.0, 1.0),
        width,
        height,
    )

    def in_time(drawn):
        if deadline is None:
            return
        now = time.monotonic()
        left = count - drawn
        if now >= deadline:
            raise OutOfTime("the run's budget was spent after %d of its %d frames" % (drawn, count))
        if drawn >= ESTIMATE_AFTER and left > 0:
            each = (now - started) / drawn
            if now + each * left > deadline:
                raise OutOfTime(
                    "its %d frames would take about %.0f s to draw, and the run's budget has %.0f s left"
                    % (count, each * count, deadline - started)
                )

    # Framed first, on everything any frame holds, so that nothing that moves
    # is re-centred from one frame to the next.
    bounds = None
    for index in range(count):
        bounds = _union(bounds, camera.bounds(_solid_points(scene_at(index))))
    camera.fit(bounds)

    # Smoothed edges where a picture is small enough for it to matter and to
    # afford; past full HD a pixel is small enough on its own.
    supersample = SUPERSAMPLE if width * height <= VIDEO_SUPERSAMPLED_PIXELS else 1
    # As many frames at once as threads help with and memory allows: each one
    # being drawn is a picture of its own.
    each = width * height * supersample * supersample * PICTURE_BYTES_PER_PIXEL
    workers = max(1, min(VIDEO_THREADS, os.cpu_count() or 1, VIDEO_MEMORY // each))
    floors = ({}, threading.Lock())
    drawers = [Frames(camera, supersample, floors) for _ in range(workers)]

    target = os.path.join(path, name)
    encoder = Encoder(target, width, height, fps, VIDEO_FORMATS[kind])
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        # In rounds of one frame for each drawer, so that no drawer is ever
        # asked for two at once and the frames are sent in order.
        for first in range(0, count, workers):
            in_time(first)
            jobs = list(zip(drawers, range(first, min(first + workers, count))))
            if pool is None:
                images = [drawer.draw(scene_at(index)) for drawer, index in jobs]
            else:
                images = list(pool.map(lambda job: job[0].draw(scene_at(job[1])), jobs))
            for image in images:
                encoder.send(image)
        encoder.close()
    except BaseException:
        encoder.abandon()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return name


class Encoder:
    """FFmpeg, turning the frames it is sent into one video file.

    The FFmpeg the ``imageio-ffmpeg`` wheel carries (or the one its
    ``IMAGEIO_FFMPEG_EXE`` names), run as a program and fed raw RGB frames on
    its standard input. Only the program is taken from that package, and not its
    writer, which never asks how FFmpeg exited: a codec this FFmpeg lacks would
    be a file that is not there, reported as written. What FFmpeg says goes to
    a file rather than to a pipe nobody reads while frames are being written,
    and a failure says the end of it.
    """

    def __init__(self, target, width, height, fps, encoding):
        import imageio_ffmpeg

        self.target = target
        self.size = (int(height), int(width), 3)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "%dx%d" % (width, height),
            "-r",
            "%g" % fps,
            "-i",
            "-",
            "-an",
            "-c:v",
            encoding["codec"],
            "-pix_fmt",
            "yuv420p",
        ]
        command += list(encoding["parameters"]) + [target]
        self.said = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.said)

    def send(self, image):
        """One frame, as (height, width, 3) bytes."""
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.shape != self.size:
            raise ValueError("a frame of %s, in a video of %s" % (image.shape, self.size))
        try:
            self.process.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError):
            self.process.wait()
            raise RuntimeError("FFmpeg stopped while it was sent the frames: %s" % self._what_it_said())

    def close(self):
        """Wait for the file to be written, or raise saying why it was not."""
        try:
            self.process.stdin.close()
        except OSError:
            pass
        status = self.process.wait()
        if status != 0 or not os.path.isfile(self.target) or os.path.getsize(self.target) == 0:
            said = self._what_it_said()
            self._remove()
            raise RuntimeError("FFmpeg did not write the video (exit status %s): %s" % (status, said))
        self.said.close()

    def abandon(self):
        """Stop FFmpeg, and leave no half-written file behind."""
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.wait()
        self.said.close()
        self._remove()

    def _remove(self):
        try:
            os.remove(self.target)
        except OSError:
            pass

    def _what_it_said(self):
        try:
            self.said.seek(0)
            text = self.said.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            text = ""
        return text[-1000:] if text else "nothing"
