#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""Pictures of a simulated scene, drawn on the CPU.

PartCAD asks every simulation for two pictures of the scene -- one before
anything moved and one when the run is over -- and says where to take them
from: ``request["snapshot"]``, read from the scene's own ``render: png:``
configuration (see 'wrappers/wrapper_simulate.py' in PartCAD itself). This is
what draws them.

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
the world for the same reason.

This file is the same in every PartCAD simulation plugin -- the MuJoCo one and
the Gazebo one -- and is kept identical rather than shared: a plugin is a
package of its own and does not reach into another. What differs between them
is only where the triangles come from.
"""

import os
import struct
import zlib

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

    def frame(self, points):
        """Fit these points into the picture, with a margin around them."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        projected = self.project(points)[:, :2]
        low, high = projected.min(axis=0), projected.max(axis=0)
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


class Picture:
    """A depth-buffered canvas, drawn into one triangle at a time."""

    def __init__(self, camera):
        self.camera = camera
        self.width = camera.width * SUPERSAMPLE
        self.height = camera.height * SUPERSAMPLE
        self.color = np.empty((self.height, self.width, 3))
        self.color[:] = BACKGROUND
        self.depth = np.full((self.height, self.width), -np.inf)
        self.ids = np.full((self.height, self.width), _BACKGROUND_ID, dtype=np.int64)
        axes = camera.axes
        self.light = _unit(axes[2] + 0.6 * axes[1] + 0.3 * axes[0])

    def shade(self, triangles, color):
        """One colour per triangle, lit by how squarely it faces the light."""
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        # Both sides: nothing says which way a face of an arbitrary mesh points.
        facing = np.abs((normals / lengths[:, None]) @ self.light)
        return np.clip(np.asarray(color, dtype=float)[None, :] * (AMBIENT + DIFFUSE * facing)[:, None], 0.0, 1.0)

    def add(self, triangles, colors, ident, bias=0.0):
        """Draw triangles (N, 3, 3) in the colours (N, 3), as the thing 'ident' names."""
        triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        if len(triangles) == 0:
            return
        projected = self.camera.project(triangles)
        xs, ys = self.camera.to_pixels(projected, SUPERSAMPLE)
        zs = projected[..., 2] - bias
        for index in range(len(triangles)):
            self._triangle(xs[index], ys[index], zs[index], colors[index], ident)

    def _triangle(self, x, y, z, color, ident):
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-12:
            return
        left = max(int(np.floor(x.min())), 0)
        right = min(int(np.ceil(x.max())), self.width - 1)
        top = max(int(np.floor(y.min())), 0)
        bottom = min(int(np.ceil(y.max())), self.height - 1)
        if left > right or top > bottom:
            return
        px = np.arange(left, right + 1) + 0.5
        py = np.arange(top, bottom + 1)[:, None] + 0.5
        # Barycentric weights, each the signed area opposite its vertex.
        w0 = ((x[1] - px) * (y[2] - py) - (x[2] - px) * (y[1] - py)) / area
        w1 = ((x[2] - px) * (y[0] - py) - (x[0] - px) * (y[2] - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return
        depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        region = self.depth[top : bottom + 1, left : right + 1]
        nearer = inside & (depth > region)
        region[nearer] = depth[nearer]
        self.color[top : bottom + 1, left : right + 1][nearer] = color
        self.ids[top : bottom + 1, left : right + 1][nearer] = ident

    def finish(self):
        """The picture at the size asked for, as (height, width, 3) bytes."""
        ids = self.ids
        edge = np.zeros(ids.shape, dtype=bool)
        # An outline wherever two things meet, unless both are the floor or
        # the background: the floor's squares are not things.
        for first, second in (
            (ids[:, 1:], ids[:, :-1]),
            (ids[1:, :], ids[:-1, :]),
        ):
            differs = (first != second) & (np.maximum(first, second) > _GROUND_ID)
            if first.shape[1] != ids.shape[1]:
                edge[:, 1:] |= differs
                edge[:, :-1] |= differs
            else:
                edge[1:, :] |= differs
                edge[:-1, :] |= differs
        color = self.color.copy()
        color[edge] *= OUTLINE
        color = color.reshape(self.camera.height, SUPERSAMPLE, self.camera.width, SUPERSAMPLE, 3).mean(axis=(1, 3))
        return (np.clip(color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


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
    theta = np.linspace(0.0, np.pi, rings + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    grid = np.stack(
        [
            rx * np.sin(theta)[:, None] * np.cos(phi)[None, :],
            ry * np.sin(theta)[:, None] * np.sin(phi)[None, :],
            rz * np.cos(theta)[:, None] * np.ones_like(phi)[None, :],
        ],
        axis=-1,
    )
    return _quads(grid)


def cylinder(radius, half_length, segments=24):
    """A capped cylinder along z, centred on the origin, as triangles."""
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    ring = np.stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)], axis=-1)
    side = _quads(np.stack([ring + (0, 0, -half_length), ring + (0, 0, half_length)]))
    caps = []
    for z in (-half_length, half_length):
        centre = np.array([0.0, 0.0, z])
        for i in range(segments):
            caps.append([centre, ring[i] + (0, 0, z), ring[i + 1] + (0, 0, z)])
    return np.concatenate([side, np.array(caps)])


def capsule(radius, half_length, segments=24, rings=12):
    """A cylinder with hemispherical ends, along z, centred on the origin."""
    # A sphere pulled apart at its equator: the upper half up, the lower down.
    ends = ellipsoid(radius, radius, radius, segments, rings)
    ends[..., 2] += np.where(ends[..., 2] >= 0, half_length, -half_length)
    side = cylinder(radius, half_length, segments)[: 2 * segments]
    return np.concatenate([ends, side])


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
    # Centred under the middle of the picture, so it reaches the edges.
    middle = camera.center[0] * camera.axes[0] + camera.center[1] * camera.axes[1]
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


def draw(camera, scene):
    """One picture of a scene: ``{"solids": [...], "planes": [...]}``.

    A solid is ``{"triangles": (N, 3, 3), "color": (r, g, b), "id": int}``, in
    world coordinates; solids sharing an id are one thing, and get no outline
    between them. A plane is ``{"origin": (3,), "axes": (3, 3), "color": ...}``,
    its normal the third column of 'axes'.
    """
    picture = Picture(camera)
    for plane in scene.get("planes") or []:
        _ground(picture, plane, _GROUND_ID)
    for solid in scene.get("solids") or []:
        triangles = np.asarray(solid["triangles"], dtype=float).reshape(-1, 3, 3)
        color = solid.get("color")
        color = DEFAULT_COLOR if color is None else color
        picture.add(triangles, picture.shade(triangles, color), _GROUND_ID + 1 + int(solid.get("id", 0)))
    return picture.finish()


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
    points = [
        np.asarray(solid["triangles"], dtype=float).reshape(-1, 3)
        for moment, scene in scenes.items()
        if moment in files
        for solid in scene.get("solids") or []
    ]
    camera.frame(np.concatenate(points) if points else np.zeros((0, 3)))

    written = {}
    for moment, scene in scenes.items():
        name = files.get(moment)
        if not name:
            continue
        write_png(os.path.join(path, name), draw(camera, scene))
        written[moment] = name
    return written
