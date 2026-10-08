#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The pictures a simulation takes of its scene (see 'snapshot_raster.py').

The renderer is tested on its own, with nothing but numpy, and then the plugin
end to end -- MuJoCo running a stack of two blocks the top one of which falls
off -- when MuJoCo is installed, which CI makes sure of.
"""

import os
import struct
import zlib

import numpy as np
import pytest

import snapshot_raster

SNAPSHOT = {
    "format": "png",
    "files": {"before": "snapshot-before.png", "after": "snapshot-after.png"},
    "width": 64,
    "height": 48,
    "viewport_origin": [100.0, -100.0, 100.0],
    "viewport_up": [0.0, 0.0, 1.0],
}


def read_png(path):
    """The pixels of an RGB PNG this module wrote, as (height, width, 3)."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8 : start - 4])[0]
    raw = zlib.decompress(data[start : start + length])
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, 1 + width * 3)
    assert not rows[:, 0].any(), "every row is written unfiltered"
    return rows[:, 1:].reshape(height, width, 3)


def cube(center, half=1.0, color=(0.8, 0.2, 0.2), ident=1):
    return {"triangles": snapshot_raster.box([half] * 3) + np.asarray(center), "color": color, "id": ident}


BACKGROUND = np.array([int(c * 255 + 0.5) for c in snapshot_raster.BACKGROUND], dtype=np.uint8)


def is_background(pixel):
    return (np.asarray(pixel) == BACKGROUND).all()


def drawn_rows(image):
    """The rows of a picture anything is drawn on."""
    return np.where((image != BACKGROUND).any(axis=2).any(axis=1))[0]


def test_the_front_view_has_x_to_the_right_and_z_up():
    """The same convention 'pc render --view front' draws in."""
    camera = snapshot_raster.Camera((0, -100, 0), (0, 0, 1), 10, 10)
    right, up, toward = camera.axes
    assert np.allclose(right, (1, 0, 0))
    assert np.allclose(up, (0, 0, 1))
    assert np.allclose(toward, (0, -1, 0))


def test_a_solid_is_drawn_in_the_middle_of_the_picture_and_nothing_is_drawn_around_it():
    camera = snapshot_raster.Camera((0, -100, 0), (0, 0, 1), 40, 40)
    scene = {"solids": [cube((5, 5, 5))]}
    camera.frame(scene["solids"][0]["triangles"])

    image = snapshot_raster.draw(camera, scene)

    assert image.shape == (40, 40, 3)
    assert not is_background(image[20, 20])
    assert is_background(image[1, 1])


def test_the_nearer_face_hides_the_farther_one():
    camera = snapshot_raster.Camera((0, -100, 0), (0, 0, 1), 40, 40)
    near = cube((0, -5, 0), color=(0.0, 0.0, 1.0), ident=1)
    far = cube((0, 5, 0), color=(1.0, 0.0, 0.0), ident=2)
    camera.frame(np.concatenate([near["triangles"], far["triangles"]]))

    image = snapshot_raster.draw(camera, {"solids": [far, near]})

    red, green, blue = image[20, 20]
    assert blue > red


def test_a_picture_is_written_as_a_png_that_reads_back_as_itself(tmp_path):
    image = (np.arange(4 * 3 * 3).reshape(4, 3, 3) * 7 % 256).astype(np.uint8)
    path = tmp_path / "x.png"

    snapshot_raster.write_png(str(path), image)

    assert (read_png(str(path)) == image).all()


def test_both_pictures_are_framed_alike_so_what_moved_is_seen_to_have_moved(tmp_path):
    """A block that fell is lower in the frame, not re-centred."""
    before = {"solids": [cube((0, 0, 10))]}
    after = {"solids": [cube((0, 0, 0))]}
    snapshot = dict(SNAPSHOT, viewport_origin=[0, -100, 0])

    written = snapshot_raster.take(str(tmp_path), snapshot, {"before": before, "after": after})

    assert written == SNAPSHOT["files"]
    rows_before = drawn_rows(read_png(str(tmp_path / "snapshot-before.png")))
    rows_after = drawn_rows(read_png(str(tmp_path / "snapshot-after.png")))
    # Picture rows grow downwards: the block in the second is further down.
    assert rows_after.mean() > rows_before.mean()


def test_the_floor_is_drawn_under_the_scene_and_does_not_decide_the_framing(tmp_path):
    plane = {"origin": np.zeros(3), "axes": np.eye(3), "color": (0.8, 0.8, 0.8)}
    scene = {"solids": [cube((0, 0, 1))], "planes": [plane]}

    snapshot_raster.take(str(tmp_path), dict(SNAPSHOT, files={"after": "a.png"}), {"after": scene})
    image = read_png(str(tmp_path / "a.png"))

    # The corners are floor rather than background: it reaches the edges.
    assert not is_background(image[0, 0])
    assert not is_background(image[-1, -1])


def test_only_png_is_drawn(tmp_path):
    with pytest.raises(ValueError, match="PNG"):
        snapshot_raster.take(str(tmp_path), dict(SNAPSHOT, format="jpeg"), {"before": {"solids": []}})


def test_every_primitive_is_a_closed_set_of_triangles():
    for triangles in (
        snapshot_raster.box((1, 2, 3)),
        snapshot_raster.ellipsoid(1, 2, 3),
        snapshot_raster.cylinder(1, 2),
        snapshot_raster.capsule(1, 2),
    ):
        assert triangles.ndim == 3 and triangles.shape[1:] == (3, 3)
        assert np.isfinite(triangles).all()


def test_an_stl_is_read_whether_it_is_binary_or_text(tmp_path):
    here = os.path.join(os.path.dirname(__file__), "tests", "data", "cube.stl")
    triangles = snapshot_raster.read_stl(here)
    assert triangles.shape[1:] == (3, 3) and len(triangles) >= 12

    text = tmp_path / "t.stl"
    text.write_text(
        "solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n"
        "   vertex 0 1 0\n  endloop\n endfacet\nendsolid t\n",
        encoding="utf-8",
    )
    assert np.allclose(snapshot_raster.read_stl(str(text)), [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])


#
# The plugin, end to end
#

FALLING = """<mujoco model="fall">
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.8 0.8 0.8 1"/>
    <body name="bottom" pos="0 0 0.01"><freejoint/>
      <geom type="box" size="0.01 0.01 0.01" rgba="0.7 0.7 0.75 1"/></body>
    <body name="top" pos="0.018 0 0.03"><freejoint/>
      <geom type="box" size="0.01 0.01 0.01" rgba="0.8 0.6 0.3 1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def falling(tmp_path):
    pytest.importorskip("mujoco")
    (tmp_path / "scene.xml").write_text(FALLING, encoding="utf-8")
    return tmp_path


def run(directory, **request):
    import simulate_mujoco

    request.setdefault("scene_file", str(directory / "scene.xml"))
    request.setdefault("duration", 3.0)
    return simulate_mujoco.process(str(directory), request)


def test_a_run_draws_what_it_was_asked_to_and_says_so(falling):
    result = run(falling, snapshot=SNAPSHOT)

    assert result["success"] is True
    assert result["snapshots"] == SNAPSHOT["files"]
    before = read_png(str(falling / "snapshot-before.png"))
    after = read_png(str(falling / "snapshot-after.png"))
    assert before.shape == after.shape == (48, 64, 3)
    # The top block fell off: the two pictures are not the same picture.
    assert (before != after).any()


def test_a_run_that_was_not_asked_for_pictures_draws_none(falling):
    result = run(falling)

    assert "snapshots" not in result
    assert not (falling / "snapshot-before.png").exists()


def test_a_picture_that_cannot_be_drawn_is_a_warning_and_not_a_failed_run(falling):
    result = run(falling, snapshot=dict(SNAPSHOT, format="jpeg"))

    assert result["success"] is True
    assert "snapshots" not in result
    assert any("snapshots could not be drawn" in warning for warning in result["warnings"])
