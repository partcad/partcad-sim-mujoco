#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The bodies and joints the MJCF exporter writes, from PartCAD's own 'kinematics'.

The trees here are the ones PartCAD itself builds: a small package of cubes and
two interfaces, a 'motion:' on one of them, composed by PartCAD's own ASSY code
-- with a stand-in for every part's geometry, so nothing needs a CAD kernel --
and handed to the exporter the way 'pc export' and 'pc sim' hand it over.

Three things are checked, each against an answer that does not come from the
exporter:

* **every row of the classification table** (section 8.8 of PartCAD's
  simulation design record) loads in MuJoCo as the joints the table says, named
  as PartCAD names them;
* **every placement** MuJoCo computes from the file -- at the joints' 'ref', and
  moved away from it -- is where PartCAD's own 'kinematics' says that
  configuration puts each body, and at 'ref' it is where the tree puts every
  part: the assembly as it is drawn;
* **physics with a closed form**: a bar on a revolute joint released horizontal
  passes the bottom at sqrt(2 m g d / I) and, swinging a little, has the period
  2 pi sqrt(I / (m g d)); a carriage on a slide falls as anything falls; a nut on
  a screw turns as its thread says.

MuJoCo is needed for all but the first few, which read the XML; those are
skipped without it, and CI installs it.
"""

import asyncio
import math
import os
import shutil
import sys
import textwrap
import xml.etree.ElementTree as ElementTree

import pytest

import partcad as pc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(pc.__file__), "wrappers"))

import kinematics  # noqa: E402
import urdf_common  # noqa: E402

CUBE_STL = os.path.join(HERE, "tests", "data", "cube.stl")
G = 9.81

# A base with a bore on top, and two parts that fit it: what every joint below
# is made of. 'bore' is where each test says how the connection moves.
RIG = """
parts:
  base:
    type: stl
    path: cube.stl
    implements:
      bore:
        top: [[0, 0, 10], [0, 0, 1], 0]
  arm:
    type: stl
    path: cube.stl
    implements:
      pin:
      bore:
        top: [[0, 0, 10], [0, 0, 1], 0]
  lid:
    type: stl
    path: cube.stl
    implements:
      pin:
interfaces:
  bore:
    ports:
      hole:
{bore}
  pin:
    ports:
      tip: [[0, 0, 0], [0.71, 0.71, 0], 180]
    mates:
      bore: {{}}
assemblies:
  rig:
    type: assy
"""

ONE_JOINT = """
links:
  - part: base
  - part: arm
    connect: {{name: base, with: pin, to: bore{params}}}
"""


def _indented(text, spaces):
    text = textwrap.dedent(text or "").strip("\n")
    return textwrap.indent(text, " " * spaces) if text else ""


@pytest.fixture
def stand_in_geometry(monkeypatch):
    """Every part a shape of no geometry, and every ASSY object composed as PartCAD composes it."""

    async def part(self, ctx):
        return dict(self.get_cache_metadata(), brep=b"BREP")

    async def assembly(self, ctx):
        await self.do_instantiate()
        return await self._get_shape_real(ctx)

    monkeypatch.setattr(pc.Part, "get_wrapped", part)
    monkeypatch.setattr(pc.Assembly, "get_wrapped", assembly)


@pytest.fixture
def export_mjcf(monkeypatch):
    """The exporter, writing every mesh as the 10 mm test cube MuJoCo can load."""
    import export_mjcf as module
    import ocp_serialize

    monkeypatch.setattr(ocp_serialize, "decode_shape", lambda obj: ("shape", obj.get("brep")))
    monkeypatch.setattr(module, "write_mesh", lambda shape, path, options: shutil.copyfile(CUBE_STL, path))
    return module


def rig(tmp_path, bore, assy=None, params=""):
    """The tree PartCAD builds for 'rig': a base and an arm connected through 'bore'."""
    root = tmp_path / "package"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(CUBE_STL, root / "cube.stl")
    (root / "partcad.yaml").write_text(RIG.format(bore=_indented(bore, 4)), encoding="utf-8")
    (root / "rig.assy").write_text(textwrap.dedent(assy or ONE_JOINT.format(params=params)), encoding="utf-8")
    ctx = pc.Context(str(root))
    assembly = ctx.get_assembly(":rig")
    assert assembly is not None
    return asyncio.run(assembly.get_wrapped(ctx))


def in_a_scene(tree, location=None, fixed=True):
    """The assembly placed in a scene, as a simulation's world places its subject."""
    subject = dict(tree, label="subject", kind=kinematics.ASSEMBLY)
    if location is not None:
        subject["location"] = location
    if fixed:
        subject["fixed"] = True
    return {"name": "//p:world", "label": "world", "kind": kinematics.SCENE, "assembly": [subject]}


def cube(mass):
    """What PartCAD hands over for the 10 mm test cube (a corner at its origin) at 'mass' kg."""
    moment = mass * 0.01**2 / 6.0
    return {
        "volume": 1000.0,
        "mass": mass,
        "centerOfMass": [5.0, 5.0, 5.0],
        "inertia": {"ixx": moment, "iyy": moment, "izz": moment, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0},
    }


def weighed(tree, physics=None, mass=0.1):
    """A properties index giving every part of the tree a mass, 'physics' by part name overriding it."""
    index = {}

    def walk(node):
        if kinematics.is_geometry(node):
            name = node.get("name")
            chosen = next((value for key, value in (physics or {}).items() if str(name).endswith(":" + key)), None)
            index[name] = {"physics": chosen if chosen is not None else cube(mass)}
        for child in node.get("assembly") or []:
            walk(child)

    walk(tree)
    return index


def export(module, tmp_path, tree, name="model.xml", **request):
    path = tmp_path / name
    request.setdefault("properties", weighed(tree))
    result = module.process(str(path), dict({"wrapped": tree}, **request))
    return result, path, ElementTree.parse(str(path)).getroot()


def mjcf_joints(mujoco_element):
    """The MJCF joints of the model's bodies, in order: an '<equality><joint>' is not one."""
    return [(joint.get("name"), joint.get("type")) for joint in mujoco_element.find("worldbody").iter("joint")]


#
# The classification table, row by row
#

ROWS = [
    # (what the bore declares, the kind, the MJCF joints written)
    ("motion: {type: revolute, limits: {lower: 0, upper: 90}}", kinematics.REVOLUTE, [("arm-base", "hinge")]),
    ("motion: continuous", kinematics.CONTINUOUS, [("arm-base", "hinge")]),
    ("motion: {type: prismatic, limits: {lower: 0, upper: 5}}", kinematics.PRISMATIC, [("arm-base", "slide")]),
    (
        "motion: cylindrical",
        kinematics.CYLINDRICAL,
        [("arm-base.turnZ", "hinge"), ("arm-base.moveZ", "slide")],
    ),
    ("threadStep: 10\nmotion: screw", kinematics.SCREW, [("arm-base.turnZ", "hinge"), ("arm-base.moveZ", "slide")]),
    # The port's X is the contact frame's Y: the half turn that makes two
    # ports face each other swaps X and Y, so that is what the implied
    # parameters are named after.
    ("motion: universal", kinematics.UNIVERSAL, [("arm-base.turnZ", "hinge"), ("arm-base.turnY", "hinge")]),
    ("motion: ball", kinematics.BALL, [("arm-base", "ball")]),
    (
        "motion: planar",
        kinematics.PLANAR,
        [("arm-base.moveY", "slide"), ("arm-base.moveX", "slide"), ("arm-base.turnZ", "hinge")],
    ),
    (
        "motion: floating",
        kinematics.FLOATING,
        [
            ("arm-base.moveY", "slide"),
            ("arm-base.moveX", "slide"),
            ("arm-base.moveZ", "slide"),
            ("arm-base.ball", "ball"),
        ],
    ),
    (
        "parameters: {turnZ: [-90, 90], moveX: [-5, 5]}\nmotion: {dof: [turnZ, moveX]}",
        kinematics.CHAIN,
        None,
    ),
]


def _row_id(row):
    return row[1]


@pytest.mark.parametrize("bore, kind, written", ROWS, ids=[_row_id(row) for row in ROWS])
def test_every_kind_of_joint_is_written_as_the_table_says(
    tmp_path, stand_in_geometry, export_mjcf, bore, kind, written
):
    tree = rig(tmp_path, bore)
    found = kinematics.bodies(tree)
    arm = found.by_name("arm")
    assert arm.joint.kind == kind

    result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    assert result["success"]
    # The root of an assembly exported on its own is welded to the world (the
    # file is for looking at), and the arm hangs off it, nested, on its joint.
    root = mujoco.find("worldbody/body[@name='rig']")
    assert root.find("freejoint") is None and root.find("joint") is None
    assert root.find("body[@name='arm']") is not None
    joints = mjcf_joints(mujoco)
    if written is not None:
        assert [(name, kind) for name, kind in joints] == written
    else:
        # A chain is its freedoms in order, one MJCF joint each.
        assert [kind for _name, kind in joints] == [
            "hinge" if freedom.kind == kinematics.TURN else "slide" for freedom in arm.joint.freedoms
        ]
    assert mujoco.find("compiler").get("angle") == "degree"


def test_a_screw_is_a_hinge_a_slide_and_the_thread_between_them(tmp_path, stand_in_geometry, export_mjcf):
    """10 mm a turn: the slide follows the hinge by 0.01 m per 2 pi rad."""
    tree = rig(tmp_path, "threadStep: 10\nmotion: screw")
    _result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    (equality,) = mujoco.findall("equality/joint")
    # The hinge follows the slide, 2 pi radians to 0.01 m: written the way round
    # MuJoCo holds it (see 'write_equalities').
    assert (equality.get("joint1"), equality.get("joint2")) == ("arm-base.turnZ", "arm-base.moveZ")
    coefficients = [float(v) for v in equality.get("polycoef").split()]
    assert coefficients[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(coefficients[1]) == pytest.approx(2 * math.pi / 0.01)
    assert coefficients[2:] == [0.0, 0.0, 0.0]


def test_a_mimic_is_an_equality_between_two_joints(tmp_path, stand_in_geometry, export_mjcf):
    assy = """
    links:
      - part: base
      - part: arm
        connect: {name: base, with: pin, to: bore, joint: drive}
      - part: lid
        connect:
          name: arm
          with: pin
          to: bore
          joint: follower
          motion: {type: revolute, mimic: {joint: drive, multiplier: -2, offset: 15}}
    """
    tree = rig(tmp_path, "motion: revolute", assy=assy)
    _result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    (equality,) = mujoco.findall("equality/joint")
    assert (equality.get("joint1"), equality.get("joint2")) == ("follower", "drive")
    a0, a1 = [float(v) for v in equality.get("polycoef").split()][:2]
    # follower = -2 drive + 15 degrees, both at nought to start: 15 degrees out.
    assert a1 == pytest.approx(-2.0)
    assert a0 == pytest.approx(math.radians(15.0))


def test_the_starting_value_is_the_ref_and_the_limits_are_the_range(tmp_path, stand_in_geometry, export_mjcf):
    """MuJoCo's zero is PartCAD's zero: the body is written where the assembly shows it, at 'ref'."""
    tree = rig(
        tmp_path,
        "parameters: {turnZ: [-90, 120]}\nmotion: revolute\nphysics: {damping: 0.05, friction: 0.2, maxEffort: 2}",
        params=", toParams: {turnZ: 30}",
    )
    result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    joint = mujoco.find(".//joint[@name='arm-base']")
    freedom = kinematics.bodies(tree).by_name("arm").joint.freedoms[0]
    assert float(joint.get("ref")) == pytest.approx(freedom.value)
    assert abs(freedom.value) == pytest.approx(30.0)
    assert joint.get("limited") == "true"
    assert sorted(float(v) for v in joint.get("range").split()) == pytest.approx(sorted([freedom.lower, freedom.upper]))
    # SI per radian, as MuJoCo takes it; the effort limit has no actuator to
    # clamp, and is reported rather than written.
    assert float(joint.get("damping")) == pytest.approx(0.05)
    assert float(joint.get("frictionloss")) == pytest.approx(0.2)
    assert joint.get("actuatorfrcrange") is None
    assert "joint maxEffort" in result["unsupported"]


def test_an_unlimited_turn_has_no_range(tmp_path, stand_in_geometry, export_mjcf):
    _result, _path, mujoco = export(export_mjcf, tmp_path, rig(tmp_path, "motion: revolute"))
    joint = mujoco.find(".//joint[@name='arm-base']")
    assert joint.get("limited") == "false"
    assert joint.get("range") is None


def test_a_free_root_is_free_to_move_only_when_asked_and_a_fixed_one_never(tmp_path, stand_in_geometry, export_mjcf):
    tree = rig(tmp_path, "motion: revolute")
    free = in_a_scene(tree, fixed=False)
    bolted = in_a_scene(tree, fixed=True)

    _result, _path, looked_at = export(export_mjcf, tmp_path, free, "a.xml")
    _result, _path, simulated = export(export_mjcf, tmp_path, free, "b.xml", static=False)
    _result, _path, fixed = export(export_mjcf, tmp_path, bolted, "c.xml", static=False)

    assert looked_at.find("worldbody/body/freejoint") is None
    assert simulated.find("worldbody/body[@name='subject']/freejoint") is not None
    assert fixed.find("worldbody/body[@name='subject']/freejoint") is None
    # The joint inside the assembly moves in all three.
    for mujoco in (looked_at, simulated, fixed):
        assert mjcf_joints(mujoco) == [("arm-base", "hinge")]


#
# Placements: where MuJoCo puts every body is where PartCAD says it is
#


def _flat_poses(tree):
    """Where the tree puts each part, composed node by node from the root: the assembly as drawn."""
    poses = {}

    def walk(node, pose):
        if kinematics.is_geometry(node):
            poses[id(node)] = pose
        for child in node.get("assembly") or []:
            walk(child, urdf_common.compose(pose, urdf_common.from_packed(child.get("location"))))

    walk(tree, urdf_common.from_packed(tree.get("location")))
    return poses


def _close_pose(a, b, tolerance=1e-9):
    (qa, ta), (qb, tb) = a, b
    same = max(abs(x - y) for x, y in zip(qa, qb))
    flipped = max(abs(x + y) for x, y in zip(qa, qb))
    return min(same, flipped) <= tolerance and max(abs(x - y) for x, y in zip(ta, tb)) <= tolerance * 1000


def _body_pose(mujoco, model, data, name):
    """A body's pose as MuJoCo computed it, in PartCAD's terms: (w x y z), millimetres."""
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index >= 0, name
    return (tuple(float(v) for v in data.xquat[index]), tuple(float(v) * 1000.0 for v in data.xpos[index]))


def _geom_pose(element):
    pos = [float(v) * 1000.0 for v in (element.get("pos") or "0 0 0").split()]
    quat = [float(v) for v in (element.get("quat") or "1 0 0 0").split()]
    return (tuple(quat), tuple(pos))


SCENE_LOCATION = [[20.0, -30.0, 500.0], [1.0, 0.0, 0.0], 90.0]


@pytest.mark.parametrize("bore, kind, written", ROWS, ids=[_row_id(row) for row in ROWS])
def test_every_part_is_where_the_assembly_shows_it_and_moves_as_partcad_says(
    tmp_path, stand_in_geometry, export_mjcf, bore, kind, written
):
    """At 'ref' every part is where the tree puts it; moved, every body is where 'kinematics' says."""
    mujoco = pytest.importorskip("mujoco")
    tree = in_a_scene(rig(tmp_path, bore, params=_STARTS.get(kind, "")), SCENE_LOCATION)
    found = kinematics.bodies(tree)
    _result, path, xml = export(export_mjcf, tmp_path, tree, static=False)

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    flat = _flat_poses(tree)
    for body in found:
        assert _close_pose(_body_pose(mujoco, model, data, body.name), body.world), body.name
        for index, part in enumerate(body.parts):
            geom = xml.find(".//geom[@name='%s_geom_%d']" % (body.name, index))
            placed = urdf_common.compose(_body_pose(mujoco, model, data, body.name), _geom_pose(geom))
            assert _close_pose(placed, flat[id(part.node)]), (body.name, part.label)

    # And moved: every scalar freedom away from where it starts, a coupled one
    # by its coupling, and the arm is where the joint's own product puts it.
    arm = found.by_name("arm")
    joint = arm.joint
    values = joint.values()
    for freedom in joint.freedoms:
        if kind == kinematics.FLOATING and freedom.kind == kinematics.TURN:
            continue
        values[freedom.index] += 17.0 if freedom.kind == kinematics.TURN else 3.0
    for freedom in joint.freedoms:
        if freedom.follows is not None:
            values[freedom.index] = freedom.ratio * values[freedom.follows] + freedom.offset

    names = [name for name, _type in mjcf_joints(xml)]
    for freedom in joint.freedoms:
        if kind in (kinematics.BALL, kinematics.FLOATING) and freedom.kind == kinematics.TURN:
            continue
        name = names[0] if len(names) == 1 else "arm-base." + export_mjcf.freedom_label(freedom)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert index >= 0, name
        scale = math.pi / 180.0 if freedom.kind == kinematics.TURN else 0.001
        data.qpos[model.jnt_qposadr[index]] = values[freedom.index] * scale
    if kind in (kinematics.BALL, kinematics.FLOATING):
        # A ball turned any which way leaves its centre where the turns meet.
        ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[-1])
        turn = urdf_common.axis_angle_to_quat((0.3, -0.5, 0.8), 40.0)
        data.qpos[model.jnt_qposadr[ball] : model.jnt_qposadr[ball] + 4] = turn
    mujoco.mj_forward(model, data)

    expected = kinematics.compose(arm.parent.world, joint.location(values))
    actual = _body_pose(mujoco, model, data, "arm")
    if kind in (kinematics.BALL, kinematics.FLOATING):
        centre = [float(v) * 1000.0 for v in xml.find(".//joint[@type='ball']").get("pos").split()]
        assert kinematics.transform_point(actual, centre) == pytest.approx(
            kinematics.transform_point(expected, centre), abs=1e-6
        )
    else:
        assert _close_pose(actual, expected), kind


# Starting values away from zero where a kind has a parameter to give one, so
# that 'ref' is exercised rather than nought.
_STARTS = {
    kinematics.REVOLUTE: ", toParams: {turnZ: 30}",
    kinematics.PRISMATIC: ", withParams: {moveZ: 3}",
    kinematics.CHAIN: ", toParams: {moveX: 2, turnZ: -20}",
}


#
# Physics with a closed form
#

# The pendulum: a bar 300 mm long and 200 g, hung from one end on the hinge.
# Its centre of mass is half-way along, and about the hinge its moment of
# inertia is a thin rod's: m L^2 / 3.
BAR_MASS, BAR_LENGTH = 0.2, 0.3
BAR_D = BAR_LENGTH / 2.0
BAR_I_PIVOT = BAR_MASS * BAR_LENGTH**2 / 3.0


def bar():
    """The bar's physics as PartCAD states them: along the arm's own X, from the hinge's axis."""
    across = BAR_MASS * BAR_LENGTH**2 / 12.0
    return {
        "volume": 1000.0,
        "mass": BAR_MASS,
        "centerOfMass": [BAR_D * 1000.0, 0.0, 0.0],
        "inertia": {"ixx": 1e-7, "iyy": across, "izz": across, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0},
    }


def pendulum(tmp_path, export_mjcf, start=0.0, duration=1.0, samples=1000):
    """The bar released at 'start' degrees from horizontal, on a revolute joint about a horizontal axis.

    The hinge is the bore's Z axis, which the scene turns to lie along world Y,
    so that the bar, along the arm's X, starts horizontal at nought.
    """
    import simulate_mujoco

    params = ", toParams: {turnZ: %s}" % start if start else ""
    tree = in_a_scene(rig(tmp_path, "motion: revolute", params=params), [[0.0, 0.0, 500.0], [1.0, 0.0, 0.0], 90.0])
    _result, path, _xml = export(
        export_mjcf, tmp_path, tree, "pendulum.xml", static=False, properties=weighed(tree, {"arm": bar()})
    )
    return simulate_mujoco.process(
        str(tmp_path), {"scene_file": str(path), "duration": duration, "samples": samples, "timestep": 0.0005}
    )


def _bottom(start_reading):
    """Which way down is, in the joint's degrees: the side the bar's weight swings it to first."""
    return 90.0 if start_reading > 0 else -90.0


def test_a_pendulum_released_horizontal_passes_the_bottom_at_the_speed_its_fall_gives_it(
    tmp_path, stand_in_geometry, export_mjcf
):
    """omega = sqrt(2 m g d / I_pivot): all of m g d turned into (1/2) I omega^2."""
    pytest.importorskip("mujoco")
    expected = math.degrees(math.sqrt(2.0 * BAR_MASS * G * BAR_D / BAR_I_PIVOT))

    result = pendulum(tmp_path, export_mjcf)

    assert result["before"]["joints"]["arm-base"]["pos"] == pytest.approx(0.0, abs=1e-9)
    readings = [sample["joints"]["arm-base"] for sample in result["samples"]]
    bottom = _bottom(readings[5]["pos"])
    fastest = max(abs(reading["vel"]) for reading in readings)
    assert fastest == pytest.approx(expected, rel=0.005)
    # And it is fastest where the bottom is.
    at_fastest = max(readings, key=lambda reading: abs(reading["vel"]))
    assert at_fastest["pos"] == pytest.approx(bottom, abs=1.0)


def test_a_pendulum_swinging_a_little_has_the_small_angle_period(tmp_path, stand_in_geometry, export_mjcf):
    """T = 2 pi sqrt(I_pivot / (m g d)), from two crossings of the bottom in the same direction."""
    pytest.importorskip("mujoco")
    expected = 2.0 * math.pi * math.sqrt(BAR_I_PIVOT / (BAR_MASS * G * BAR_D))

    # Released 5 degrees short of the bottom, which a first run says the side of.
    probe = pendulum(tmp_path / "probe", export_mjcf, duration=0.05, samples=10)
    bottom = _bottom(probe["samples"][-1]["joints"]["arm-base"]["pos"])
    result = pendulum(tmp_path, export_mjcf, start=bottom - math.copysign(5.0, bottom), duration=2.0, samples=2000)

    crossings = []
    previous = None
    for sample in result["samples"]:
        offset = sample["joints"]["arm-base"]["pos"] - bottom
        if previous is not None and previous[1] < 0.0 <= offset:
            # Linearly between the two samples either side of it.
            t0, o0 = previous
            crossings.append(t0 + (sample["time"] - t0) * (-o0) / (offset - o0))
        previous = (sample["time"], offset)
    assert len(crossings) >= 2
    assert crossings[1] - crossings[0] == pytest.approx(expected, rel=0.003)


def test_a_carriage_on_a_slide_falls_as_anything_falls(tmp_path, stand_in_geometry, export_mjcf):
    """Half g t squared along the slide, and not a millimetre across it."""
    pytest.importorskip("mujoco")
    import simulate_mujoco

    tree = in_a_scene(rig(tmp_path, "motion: prismatic"), [[0.0, 0.0, 800.0], [0.0, 0.0, 1.0], 0.0])
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False)
    result = simulate_mujoco.process(str(tmp_path), {"scene_file": str(path), "duration": 0.3, "timestep": 0.0005})

    before, after = result["before"], result["after"]
    fallen = 0.5 * G * after["time"] ** 2 * 1000.0
    assert after["joints"]["arm-base"]["type"] == "prismatic"
    assert abs(after["joints"]["arm-base"]["pos"] - before["joints"]["arm-base"]["pos"]) == pytest.approx(
        fallen, rel=0.01
    )
    drop = [a - b for a, b in zip(after["bodies"]["arm"]["pos"], before["bodies"]["arm"]["pos"])]
    assert drop[2] == pytest.approx(-fallen, rel=0.01)
    assert abs(drop[0]) < 1e-6 and abs(drop[1]) < 1e-6


def test_a_nut_on_a_screw_turns_as_its_thread_says(tmp_path, stand_in_geometry, export_mjcf):
    """Pulled down its axis by its weight, the nut turns 360 degrees for every 10 mm it moves."""
    pytest.importorskip("mujoco")
    import simulate_mujoco

    tree = in_a_scene(rig(tmp_path, "threadStep: 10\nmotion: screw"), [[0.0, 0.0, 800.0], [0.0, 0.0, 1.0], 0.0])
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False)
    result = simulate_mujoco.process(
        str(tmp_path), {"scene_file": str(path), "duration": 0.5, "samples": 20, "timestep": 0.0005}
    )

    move = kinematics.bodies(tree).by_name("arm").joint.freedoms[1]
    after = result["after"]["joints"]
    moved = after["arm-base.moveZ"]["pos"]
    assert abs(moved) > 20.0
    for reading in [sample["joints"] for sample in result["samples"]] + [after]:
        turned = reading["arm-base.turnZ"]["pos"]
        assert reading["arm-base.moveZ"]["pos"] == pytest.approx(move.ratio * turned + move.offset, abs=0.05)
