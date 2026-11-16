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
    # And the right way round: a corner drawn where the hill and the slope put PartCAD's surface, which a
    # mirror of the rows in X or in Y would move. The geom stands at the middle of the rectangle, at its
    # lowest height.
    gx, gy, gz = (float(v) * 1000.0 for v in model.geom_pos[0])
    relief = float(numpy.max(poles) - numpy.min(poles))
    for triangle in triangles[:: max(1, len(triangles) // 25)]:
        px, py, pz = (float(v) * 1000.0 for v in triangle[0])
        expected = heightfield.evaluate(poles, info["extent"], gx + px, gy + py)
        assert gz + pz == pytest.approx(expected, abs=0.02 * relief), (px, py)


# -- What is not one, however it looks -------------------------------------------------------------------------


def test_without_occt_nothing_is_one(monkeypatch):
    # A real elevation map, so that only OCCT missing can make it None - as on
    # a CI with the other tests' mocked shapes and no OCCT. Every OCP module
    # this process has loaded is hidden, not only the package: a loaded one is
    # found without its parent being asked.
    shape, _, _ = built()
    for name in [name for name in sys.modules if name == "OCP" or name.startswith("OCP.")]:
        monkeypatch.setitem(sys.modules, name, None)
    assert export_mjcf.elevation_map(shape) is None


def test_a_hole_through_it_is_not_one():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    shape, info, _ = built()
    x0, y0, x1, y1 = info["extent"]
    # A shaft through the middle: the top's bounds are the same, and it is not the whole surface.
    shaft = BRepPrimAPI_MakeBox(gp_Pnt((x0 + x1) / 2 - 500.0, (y0 + y1) / 2 - 500.0, -1000.0), 1000.0, 1000.0, 20000.0)
    holed = BRepAlgoAPI_Cut(shape, shaft.Shape()).Shape()
    assert export_mjcf.elevation_map(shape) is not None
    assert export_mjcf.elevation_map(holed) is None


def test_a_coordinate_is_linear_where_the_poles_are_at_their_greville_places():
    knots = [0.0] * 4 + [0.25, 0.5, 0.75] + [1.0] * 4
    places = export_mjcf.greville(knots, 3)
    assert places == pytest.approx([0.0, 1 / 12.0, 0.25, 0.5, 0.75, 11 / 12.0, 1.0])
    assert export_mjcf.linear_in_parameter([100.0 + 1200.0 * g for g in places], knots, 3, 100.0, 1300.0)
    # Evenly spaced poles of a cubic: the same at both ends and at every knot's middle, and bent between.
    even = [100.0 + 1200.0 * k / 6.0 for k in range(7)]
    assert not export_mjcf.linear_in_parameter(even, knots, 3, 100.0, 1300.0)
    # A degree-1 surface's poles stand at its knots: at a quarter, a pole a quarter of the way along, or a bend.
    knots = [0.0, 0.0, 0.25, 1.0, 1.0]
    assert export_mjcf.linear_in_parameter([0.0, 10.0, 40.0], knots, 1, 0.0, 40.0)
    assert not export_mjcf.linear_in_parameter([0.0, 20.0, 40.0], knots, 1, 0.0, 40.0)


def test_what_an_export_says_it_wrote_is_there(tmp_path):
    shape, _, _ = built()
    path = tmp_path / "site.xml"
    result = export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object()})
    assert result["success"], result
    # A height field is written inline: it is no mesh file, and nothing says it is.
    assert result["meshes"] == []
    assert all(os.path.isfile(os.path.join(str(tmp_path), name)) for name in result["files"]), result["files"]
    assert not any("Nothing was exported" in warning for warning in result["warnings"])


def test_a_ground_that_moves_keeps_its_mesh(tmp_path):
    shape, _, _ = built()
    path = tmp_path / "site.xml"
    result = export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object(), "static": False})
    assert result["success"], result
    model = ElementTree.parse(str(path)).getroot()
    # Free to fall, it would fall through the ground as a height field: it is its mesh, and said to be.
    assert not model.find("asset").findall("hfield") and model.find("asset").findall("mesh")
    assert any("moves" in warning for warning in result["warnings"])
    assert all(os.path.isfile(os.path.join(str(tmp_path), name)) for name in result["files"])


# -- Reading it back ----------------------------------------------------------------------------------------------


def test_a_landscape_to_be_read_back_is_exported_as_its_mesh(tmp_path):
    """A height field is for MuJoCo and is not read back (see the README): 'hfield: false' is the round trip."""
    import import_mjcf

    shape, _, _ = built()
    as_field, as_mesh = tmp_path / "field" / "site.xml", tmp_path / "mesh" / "site.xml"
    for path, hfield in ((as_field, True), (as_mesh, False)):
        path.parent.mkdir()
        result = export_mjcf.process(str(path), {"shape_tree": scene_of(shape), "wrapped": object(), "hfield": hfield})
        assert result["success"], result

    back = import_mjcf.process(None, {"source_file": str(as_mesh), "output_folder": str(tmp_path / "mesh")})
    parts = [link for link in back["root"]["links"] if link.get("type") == "part"]
    assert len(parts) == 1 and parts[0]["part_file"].endswith(".stl") and os.path.isfile(parts[0]["part_file"])

    # The height field is not read back, and a model of nothing else is no model to import.
    with pytest.raises(ValueError, match="No geometry found"):
        import_mjcf.process(None, {"source_file": str(as_field), "output_folder": str(tmp_path / "field")})
