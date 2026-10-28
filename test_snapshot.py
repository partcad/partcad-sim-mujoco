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
import time
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


def test_the_floor_is_under_a_scene_far_from_the_origin_too(tmp_path):
    """The floor is centred where the line of sight through the middle of the
    picture meets it, not where that line crosses the plane through the origin
    -- which, for a scene a long way from the origin, is off the picture."""
    plane = {"origin": np.zeros(3), "axes": np.eye(3), "color": (0.8, 0.8, 0.8)}
    scene = {"solids": [cube((1000, 0, 1))], "planes": [plane]}

    snapshot_raster.take(str(tmp_path), dict(SNAPSHOT, files={"after": "a.png"}), {"after": scene})
    image = read_png(str(tmp_path / "a.png"))

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


def reference(camera, triangles, supersample=snapshot_raster.SUPERSAMPLE):
    """Which triangle is nearest at every pixel, worked out one pixel at a time: -1 for none.

    The plainest way there is of filling triangles, to hold the batched one to.
    """
    triangles = np.asarray(triangles, dtype=float)
    projected = camera.project(triangles)
    xs, ys = camera.to_pixels(projected, supersample)
    zs = projected[..., 2]
    height, width = camera.height * supersample, camera.width * supersample
    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    nearest = np.full((height, width), -np.inf)
    which = np.full((height, width), -1)
    for index, (x, y, z) in enumerate(zip(xs, ys, zs)):
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        w0 = ((x[1] - px) * (y[2] - py) - (x[2] - px) * (y[1] - py)) / area
        w1 = ((x[2] - px) * (y[0] - py) - (x[0] - px) * (y[2] - py)) / area
        w2 = 1.0 - w0 - w1
        depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        nearer = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (depth > nearest)
        nearest[nearer] = depth[nearer]
        which[nearer] = index
    return which


def test_triangles_of_every_size_are_filled_as_one_at_a_time_would_fill_them():
    """The batched fill, the large-triangle fill, and the two together, against the plainest one."""
    rng = np.random.default_rng(7)
    camera = snapshot_raster.Camera((0, 0, 100), (0, 1, 0), 120, 90)
    camera.frame(np.array([[-60.0, -45.0, 0.0], [60.0, 45.0, 0.0]]))
    centres = rng.uniform(-60, 60, (300, 1, 3))
    # Mostly a few pixels across, some long and thin, a few larger than a batch.
    sizes = np.concatenate([rng.uniform(0.5, 4, 240), rng.uniform(10, 40, 50), rng.uniform(40, 90, 10)])
    triangles = centres + rng.normal(0, 1, (300, 3, 3)) * sizes[:, None, None]
    triangles[200:250, :, 1] *= 0.05

    one_at_a_time = snapshot_raster.Picture(camera)
    for index, triangle in enumerate(triangles):
        # One call each, as a thing of its own, so that the ids say which won.
        one_at_a_time.add(triangle[None], np.zeros((1, 3)), index)
    batched = snapshot_raster.Picture(camera)
    for start in range(0, len(triangles), 100):
        # A hundred in one call, which is what batches them; each one's index
        # is its colour, which says which won -- exactly, over a power of two.
        indices = np.arange(start, start + 100)
        colors = np.stack([indices / 512.0, np.zeros(100), np.zeros(100)], axis=1)
        batched.add(triangles[start : start + 100], colors, 1)

    expected = reference(camera, triangles)
    background = snapshot_raster._BACKGROUND_ID
    alone = np.where(one_at_a_time.ids == background, -1, one_at_a_time.ids)
    together = np.where(batched.ids == background, -1, np.round(batched.color[..., 0] * 512).astype(int))
    # A pixel centre exactly on an edge two triangles share can go either way.
    assert (alone != expected).mean() < 1e-3
    assert (together != expected).mean() < 1e-3


def test_a_closed_surface_is_turned_outward_and_an_open_one_is_left_as_it_is():
    for triangles in (
        snapshot_raster.box((1, 2, 3)),
        snapshot_raster.ellipsoid(1, 2, 3),
        snapshot_raster.cylinder(1, 2),
        snapshot_raster.capsule(1, 2),
        snapshot_raster.capsule(1, 2, rings=7),
    ):
        for given in (triangles, triangles[:, ::-1]):
            outward, closed = snapshot_raster.outward(given)
            assert closed
            volume = np.einsum("ij,ij->i", outward[:, 0], np.cross(outward[:, 1], outward[:, 2])).sum()
            assert volume > 0

    open_box = snapshot_raster.box((1, 1, 1))[:-1]
    assert snapshot_raster.outward(open_box)[1] is False
    mixed = snapshot_raster.box((1, 1, 1)).copy()
    mixed[0] = mixed[0][::-1]
    assert snapshot_raster.outward(mixed)[1] is False


def test_the_back_of_a_closed_surface_is_not_drawn_and_nothing_shows_it():
    camera = snapshot_raster.Camera((100, -100, 100), (0, 0, 1), 64, 48)
    triangles, closed = snapshot_raster.outward(snapshot_raster.ellipsoid(10, 6, 4))
    assert closed
    camera.frame(triangles)
    solid = {"triangles": triangles, "color": (0.8, 0.4, 0.2), "id": 1}

    every = snapshot_raster.draw(camera, {"solids": [solid]})
    culled = snapshot_raster.draw(camera, {"solids": [dict(solid, closed=True)]})

    assert (every == culled).all()


def test_every_frame_of_a_video_is_the_picture_drawn_afresh():
    """What 'Frames' draws again of a frame, and copies of the floor, adds up to the whole picture."""
    plane = {"origin": np.zeros(3), "axes": np.eye(3), "color": (0.8, 0.8, 0.8)}
    scenes = [
        {"planes": [plane], "solids": [cube((step * 3.0, 0, 10 - step)), cube((0, 5, 1), ident=2)]} for step in range(8)
    ]
    camera = snapshot_raster.Camera((100, -100, 100), (0, 0, 1), 64, 48)
    camera.frame(np.concatenate([snapshot_raster._solid_points(scene) for scene in scenes]))
    frames = snapshot_raster.Frames(camera)

    for scene in scenes:
        assert (frames.draw(scene) == snapshot_raster.draw(camera, scene)).all()


def test_a_video_has_a_frame_for_every_frame_time_of_the_run():
    assert snapshot_raster.frame_times(10.0, 25) == [index / 25.0 for index in range(251)]
    assert len(snapshot_raster.frame_times(0.1, 25)) == 3
    assert snapshot_raster.frame_times(0.0, 25) == [0.0]
    assert snapshot_raster.frame_times(1.0, 0) == []


VIDEO = {
    "format": "webm",
    "file": "video.webm",
    "fps": 25,
    "width": 64,
    "height": 48,
    "viewport_origin": [100.0, -100.0, 100.0],
    "viewport_up": [0.0, 0.0, 1.0],
}


def falling_cube(index):
    plane = {"origin": np.zeros(3), "axes": np.eye(3), "color": (0.8, 0.8, 0.8)}
    return {"planes": [plane], "solids": [cube((0, 0, 10 - index))]}


def test_a_video_is_recorded_as_webm(tmp_path):
    pytest.importorskip("imageio_ffmpeg")

    written = snapshot_raster.record(str(tmp_path), VIDEO, 10, falling_cube)

    assert written == "video.webm"
    with open(tmp_path / "video.webm", "rb") as f:
        # The EBML header every Matroska file, and so every WebM, starts with.
        assert f.read(4) == b"\x1a\x45\xdf\xa3"


def test_only_webm_is_recorded(tmp_path):
    with pytest.raises(ValueError, match="webm"):
        snapshot_raster.record(str(tmp_path), dict(VIDEO, format="mp4"), 10, falling_cube)


def test_a_video_that_would_not_be_finished_in_time_is_given_up_and_leaves_no_file(tmp_path):
    pytest.importorskip("imageio_ffmpeg")

    with pytest.raises(snapshot_raster.OutOfTime):
        snapshot_raster.record(str(tmp_path), VIDEO, 10, falling_cube, deadline=time.monotonic() - 1)

    assert not (tmp_path / "video.webm").exists()


def test_a_video_ffmpeg_cannot_write_says_what_ffmpeg_said(tmp_path, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setitem(snapshot_raster.VIDEO_FORMATS, "webm", {"codec": "no-such-codec", "parameters": []})

    with pytest.raises(RuntimeError, match="FFmpeg"):
        snapshot_raster.record(str(tmp_path), VIDEO, 10, falling_cube)

    assert not (tmp_path / "video.webm").exists()


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


def test_a_run_records_the_video_it_was_asked_for_and_says_so(falling):
    pytest.importorskip("imageio_ffmpeg")

    result = run(falling, video=VIDEO)

    assert result["success"] is True
    assert result["video"] == "video.webm"
    assert (falling / "video.webm").stat().st_size > 0


def test_a_run_that_was_not_asked_for_a_video_records_none(falling):
    result = run(falling)

    assert "video" not in result
    assert not (falling / "video.webm").exists()


def test_a_video_that_cannot_be_recorded_is_a_warning_and_not_a_failed_run(falling):
    result = run(falling, video=dict(VIDEO, format="mp4"))

    assert result["success"] is True
    assert "video" not in result
    assert any("video could not be recorded" in warning for warning in result["warnings"])


def test_the_film_keeps_a_frame_for_every_frame_time_however_long_the_step(falling):
    import mujoco
    import simulate_mujoco

    model = mujoco.MjModel.from_xml_path(str(falling / "scene.xml"))
    model.opt.timestep = 0.1
    data = mujoco.MjData(model)
    film = simulate_mujoco.Film(mujoco, model, VIDEO, 1.0, [])
    film.watch(data)
    while data.time < 1.0:
        mujoco.mj_step(model, data)
        film.watch(data)

    # A step of a tenth of a second lasts two and a half frames.
    assert len(film.poses) == len(snapshot_raster.frame_times(1.0, 25)) == 26
