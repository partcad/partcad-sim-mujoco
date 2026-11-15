#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""An elevation map exported as MuJoCo's height field, and nothing else exported as one.

MuJoCo collides with a mesh as its convex hull, which fills every valley of a
landscape; a height field it collides with as the surface it is. So a part that
is definitely an elevation map - by its geometry, as PartCAD's 'heightmap' and
'pointcloud' types build one - is written as an '<hfield>', and what is held to
that here is that it lands where the part is: the heights MuJoCo's own ray finds
on the model are the heights of the surface PartCAD built, row order and all.

It needs PartCAD with the height-field builder of its '//builtin/import'
('heightfield.py', which the 'heightmap' type builds with) and OCCT; MuJoCo for
the last check, which is skipped without it.
"""

import math
import os
import sys
import xml.etree.ElementTree as ElementTree

import pytest

import partcad as pc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(pc.__file__), "wrappers"))

import export_mjcf  # noqa: E402

BUILTIN_IMPORT = os.path.join(os.path.dirname(pc.__file__), "builtin", "import")
if not os.path.isfile(os.path.join(BUILTIN_IMPORT, "heightfield.py")):
    pytest.skip("this PartCAD has no height-field builder ('heightmap' part type)", allow_module_level=True)
sys.path.append(BUILTIN_IMPORT)

import heightfield  # noqa: E402
import numpy  # noqa: E402


def hill():
    """A hill off-centre and a slope, so that a row order or a sign wrong shows."""
    x, y = numpy.meshgrid(numpy.linspace(-1, 1, 40), numpy.linspace(-1, 1, 24), indexing="ij")
    return 3000.0 * numpy.exp(-((x - 0.4) ** 2 + (y + 0.3) ** 2) * 4) + 500.0 * (y + 1)


def built():
    heights = hill()
    shape, info = heightfield.build(heights, 250.0, 400.0)
    poles = heightfield.poles(400.0 + heights - heights.min(), *info["poles"])
    return shape, info, poles


def scene_of(shape, location=None):
    node = {"name": "//p:ground", "label": "ground", "brep": b"GROUND", "shape": shape}
    if location is not None:
        node["location"] = location
    return {"name": "//p:site", "label": "site", "kind": "scene", "assembly": [node]}


def test_an_elevation_map_is_found_by_its_geometry():
    shape, info, poles = built()
    field = export_mjcf.elevation_map(shape)
    assert field is not None
    assert field["extent"] == pytest.approx(info["extent"], abs=1e-6)
    assert field["base"] == pytest.approx(0.0, abs=1e-6)
    heights = field["heights"]
    ncol, nrow = len(heights), len(heights[0])
    x0, y0, x1, y1 = info["extent"]
    for i, j in [(0, 0), (ncol - 1, 0), (0, nrow - 1), (ncol // 2, nrow // 3)]:
        x = x0 + (x1 - x0) * i / (ncol - 1)
        y = y0 + (y1 - y0) * j / (nrow - 1)
        assert heights[i][j] == pytest.approx(heightfield.evaluate(poles, info["extent"], x, y), abs=1e-6)


def test_nothing_else_is_one():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

    assert export_mjcf.elevation_map(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()) is None
    assert export_mjcf.elevation_map(BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()) is None


def test_it_is_written_as_a_height_field_where_the_part_is(tmp_path):
    shape, info, _ = built()
    path = tmp_path / "site.xml"
    result = export_mjcf.process(
        str(path), {"shape_tree": scene_of(shape, [[1000.0, 0.0, 0.0], [0, 0, 1], 0]), "wrapped": object()}
    )
    assert result["success"], result
    model = ElementTree.parse(str(path)).getroot()
    fields = model.find("asset").findall("hfield")
    assert len(fields) == 1 and not model.find("asset").findall("mesh")
    nrow, ncol = int(fields[0].get("nrow")), int(fields[0].get("ncol"))
    assert len(fields[0].get("elevation").split()) == nrow * ncol
    half_x, half_y, rise, depth = (float(v) for v in fields[0].get("size").split())
    x0, y0, x1, y1 = info["extent"]
    assert (half_x, half_y) == pytest.approx(((x1 - x0) / 2000.0, (y1 - y0) / 2000.0))
    body = next(b for b in model.iter("body") if any(g.get("type") == "hfield" for g in b.findall("geom")))
    geom = next(g for g in body.findall("geom") if g.get("type") == "hfield")
    assert geom.get("hfield") == fields[0].get("name")
    lowest = export_mjcf.elevation_map(shape)["lowest"]
    # The element is a body where the scene places it; the field sits in it at the middle of its rectangle,
    # at its lowest height.
    assert [float(v) for v in body.get("pos").split()] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert [float(v) for v in geom.get("pos").split()] == pytest.approx(
        [(x0 + x1) / 2000.0, (y0 + y1) / 2000.0, lowest / 1000.0], abs=1e-9
    )
    assert depth == pytest.approx(lowest / 1000.0)


def test_hfield_false_writes_the_mesh(tmp_path):
    shape, _, _ = built()
    path = tmp_path / "site.xml"
    result = export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object(), "hfield": False})
    assert result["success"], result
    model = ElementTree.parse(str(path)).getroot()
    assert not model.find("asset").findall("hfield") and model.find("asset").findall("mesh")


def test_mujoco_finds_the_ground_where_partcad_built_it(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    shape, info, poles = built()
    path = tmp_path / "site.xml"
    assert export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object(), "ground_plane": False})[
        "success"
    ]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = numpy.ones(6, dtype=numpy.uint8)
    found = numpy.array([-1], dtype=numpy.int32)
    x0, y0, x1, y1 = info["extent"]
    relief = float(numpy.max(poles) - numpy.min(poles))
    for fx, fy in [(0.5, 0.5), (0.7, 0.35), (0.1, 0.9), (0.95, 0.05), (0.3, 0.6)]:
        x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        top = 100.0
        distance = mujoco.mj_ray(
            model, data, numpy.array([x / 1000.0, y / 1000.0, top]), numpy.array([0.0, 0.0, -1.0]), groups, 1, -1, found
        )
        assert distance > 0, (fx, fy)
        ground = (top - distance) * 1000.0
        # MuJoCo is bilinear between its samples, PartCAD's surface cubic: they meet at the samples.
        assert ground == pytest.approx(heightfield.evaluate(poles, info["extent"], x, y), abs=0.02 * relief), (fx, fy)
    assert not math.isnan(float(model.hfield_size[0][2]))


def test_a_snapshot_draws_the_height_field_mujoco_compiled(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    import simulate_mujoco

    shape, info, poles = built()
    path = tmp_path / "site.xml"
    assert export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object(), "ground_plane": False})[
        "success"
    ]
    model = mujoco.MjModel.from_xml_path(str(path))
    triangles = simulate_mujoco.heightfield_triangles(model, 0)
    x0, y0, x1, y1 = info["extent"]
    assert triangles.shape[1:] == (3, 3)
    # In the geom's frame: the rectangle about its middle, from its lowest height up by its rise.
    assert triangles[..., 0].min() == pytest.approx(-(x1 - x0) / 2000.0)
    assert triangles[..., 1].max() == pytest.approx((y1 - y0) / 2000.0)
    assert triangles[..., 2].min() == pytest.approx(0.0, abs=1e-9)
    assert triangles[..., 2].max() == pytest.approx(float(model.hfield_size[0][2]), rel=1e-6)
