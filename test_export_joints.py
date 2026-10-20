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

Four things are checked, each against an answer that does not come from the
exporter:

* **every row of the translation table** (section 8.13.4 of PartCAD's
  simulation design record) loads in MuJoCo as the joints the table says, named
  as PartCAD names them, with its bounds in MuJoCo's own terms (8.13.6): an
  open side infinite, a ball's one cone;
* **every placement** MuJoCo computes from the file, at its start, is where the
  tree puts every part -- the assembly as it is drawn, every joint started
  away from its zero, a bounded ball through the model's keyframe;
* **every reading round trip**: MuJoCo's values of a joint's MJCF joints,
  moved anywhere, read back through the joint map in the sidecar
  ('kinematics.reading()') as PartCAD's values, are where PartCAD's own
  'joint.location()' puts the body -- to 1e-9 m, for every kind, a re-zeroed
  ball and a floating body included -- and a floating joint's rates are how
  fast its readings change;
* **physics with a closed form**: a bar on a revolute joint released horizontal
  passes the bottom at sqrt(2 m g d / I) and, swinging a little, has the period
  2 pi sqrt(I / (m g d)); a carriage on a slide falls as anything falls; a nut on
  a screw turns as its thread says; a block on a floating joint falls as
  anything falls and lands *on* its base; a bar on a ball bounded by a cone
  stays inside it, about PartCAD's zero.

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
import naming  # noqa: E402
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
    monkeypatch.setattr(module, "write_mesh", mesh_copied_from(CUBE_STL))
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
    tree = asyncio.run(assembly.get_wrapped(ctx))
    # A rig that starts a joint against its own stop is a rig nothing here
    # could be checked with: MuJoCo throws the joint back in on its first step.
    # PartCAD's 'connect' check fails it, and the exporter would only warn.
    outside = [problem for problem in kinematics.bodies(tree).problems if "outside its bounds" in problem]
    assert not outside, outside
    return tree


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


def mesh_copied_from(source):
    """A stand-in for 'export_mjcf.write_mesh': the mesh is a copy of 'source', written at the tolerance asked for."""

    def write_mesh(shape, path, options):
        shutil.copyfile(source, path)
        return options["tolerance"], 12

    return write_mesh


def export(module, tmp_path, tree, name="model.xml", **request):
    path = tmp_path / name
    request.setdefault("properties", weighed(tree))
    result = module.process(str(path), dict({"wrapped": tree}, **request))
    return result, path, ElementTree.parse(str(path)).getroot()


def mjcf_joints(mujoco_element):
    """The MJCF joints of the model's bodies, in order, as (name, type): an '<equality><joint>' is not one.

    A floating joint's '<freejoint>' is one, named as PartCAD names the joint,
    and typed 'free'. The unnamed '<freejoint>' a free root of a scene is given
    is no PartCAD joint, and is left out.
    """
    found = []
    for element in mujoco_element.find("worldbody").iter():
        if element.tag == "joint":
            found.append((element.get("name"), element.get("type")))
        elif element.tag == "freejoint" and element.get("name"):
            found.append((element.get("name"), "free"))
    return found


def sidecar_names(path):
    """What a run reads the model's names and joint maps from: the sidecar beside it."""
    import simulate_mujoco

    return simulate_mujoco.Names.beside(str(path))


def started(mujoco, model):
    """MuJoCo's state where a run starts it: at the model's own start, or at its keyframe where it has one."""
    import simulate_mujoco

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, simulate_mujoco.START_KEY)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def read(mujoco, model, data, path):
    """Every joint as a run reports it, through the sidecar beside 'path'."""
    import simulate_mujoco

    return simulate_mujoco.joints(mujoco, model, data, mujoco.MjData(model), sidecar_names(path))


def values_of(joint, reading):
    """A reading's 'pos' as the joint's values, by index: a number for one freedom, by parameter for several."""
    names = joint.parameter_names()
    pos = reading["pos"]
    if not isinstance(pos, dict):
        assert len(names) == 1
        pos = {names[0]: pos}
    return [pos[name] for name in names]


#
# The translation table, row by row
#

# (id, what the bore declares, the kind, the MJCF joints written, a start)
#
# The start is what a placement and a reading are checked from: away from zero
# wherever the kind has a parameter to give one (section 8.13.7), so that every
# 'ref', every re-zeroing and every keyframe is exercised rather than nought.
# Note that 'limits:' is stated about the motion's own axis and the start about
# the parameter, and in this rig the two run opposite ways (the pin is turned
# over to face the bore): 'limits: {lower: 0, upper: 90}' is 'turnZ' in [-90, 0].
ROWS = [
    (
        "revolute",
        "motion: {type: revolute, limits: {lower: 0, upper: 90}}",
        kinematics.REVOLUTE,
        [("arm-base", "hinge")],
        ", toParams: {turnZ: -30}",
    ),
    # 'continuous' is a spelling of a revolute with both sides open: wound
    # three quarters of a turn, which a reading wrapped into +-180 would lose.
    ("continuous", "motion: continuous", kinematics.REVOLUTE, [("arm-base", "hinge")], ", toParams: {turnZ: 270}"),
    # Bounded on one side only, the door's: 'turnZ' from -30 up, and open.
    (
        "door",
        "motion: {type: revolute, limits: {upper: 30}}",
        kinematics.REVOLUTE,
        [("arm-base", "hinge")],
        ", toParams: {turnZ: 45}",
    ),
    (
        "prismatic",
        "motion: {type: prismatic, limits: {lower: 0, upper: 5}}",
        kinematics.PRISMATIC,
        [("arm-base", "slide")],
        ", toParams: {moveZ: -3}",
    ),
    (
        "incline",
        "motion: {type: prismatic, limits: {lower: 0}}",
        kinematics.PRISMATIC,
        [("arm-base", "slide")],
        ", toParams: {moveZ: -20}",
    ),
    (
        "cylindrical",
        "motion: cylindrical",
        kinematics.CYLINDRICAL,
        [("arm-base.turnZ", "hinge"), ("arm-base.moveZ", "slide")],
        ", toParams: {turnZ: 5, moveZ: 2}",
    ),
    (
        "screw",
        "threadStep: 10\nmotion: screw",
        kinematics.SCREW,
        [("arm-base.turnZ", "hinge"), ("arm-base.moveZ", "slide")],
        ", toParams: {turnZ: 36}",
    ),
    # The port's X is the contact frame's Y: the half turn that makes two
    # ports face each other swaps X and Y, so that is what the implied
    # parameters are named after.
    (
        "universal",
        "motion: universal",
        kinematics.UNIVERSAL,
        [("arm-base.turnZ", "hinge"), ("arm-base.turnY", "hinge")],
        ", toParams: {turnZ: 5, turnY: 10}",
    ),
    # A ball with no bound is re-zeroed at its start; one bounded at its zero
    # is the same ball with a cone; one bounded and started away from its zero
    # is written at its zero and started from the keyframe.
    ("ball", "motion: ball", kinematics.BALL, [("arm-base", "ball")], ", toParams: {turnX: 10, turnY: 20}"),
    ("cone", "motion: {type: ball, limits: {upper: 20}}", kinematics.BALL, [("arm-base", "ball")], ""),
    (
        "cone_keyed",
        "motion: {type: ball, limits: {upper: 20}}",
        kinematics.BALL,
        [("arm-base", "ball")],
        ", toParams: {turnX: 15, turnZ: 30}",
    ),
    (
        "planar",
        "motion: planar",
        kinematics.PLANAR,
        [("arm-base.moveY", "slide"), ("arm-base.moveX", "slide"), ("arm-base.turnZ", "hinge")],
        ", toParams: {moveX: 5, moveY: 7, turnZ: 30}",
    ),
    # The same pair in other coordinates: the turn first, then the two moves
    # across it, which is planar too and written in the order declared.
    (
        "planar_turn_first",
        "motion: {dof: [turnZ, moveX, moveY]}",
        kinematics.PLANAR,
        [("arm-base.turnZ", "hinge"), ("arm-base.moveX", "slide"), ("arm-base.moveY", "slide")],
        ", toParams: {turnZ: 30, moveX: 5, moveY: 7}",
    ),
    # A free body of its own, directly in the '<worldbody>', and no slides.
    (
        "floating",
        "motion: floating",
        kinematics.FLOATING,
        [("arm-base", "free")],
        ", toParams: {turnZ: 20, moveZ: -30}",
    ),
    (
        "composite",
        "parameters: {turnZ: [-90, 90], moveX: [-5, 5]}\nmotion: {dof: [turnZ, moveX]}",
        kinematics.COMPOSITE,
        None,
        ", toParams: {moveX: 2, turnZ: -20}",
    ),
]
ROW_IDS = [row[0] for row in ROWS]


@pytest.mark.parametrize("row, bore, kind, written, start", ROWS, ids=ROW_IDS)
def test_every_kind_of_joint_is_written_as_the_table_says(
    tmp_path, stand_in_geometry, export_mjcf, row, bore, kind, written, start
):
    tree = rig(tmp_path, bore)
    found = kinematics.bodies(tree)
    arm = found.by_name("arm")
    assert arm.joint.kind == kind

    result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    assert result["success"]
    # The root of an assembly exported on its own is welded to the world (the
    # file is for looking at), and the arm hangs off it, nested, on its joint
    # -- or, floating, is a body of its own beside it, free whatever 'static'
    # says, because the assembly says it is loose.
    root = mujoco.find("worldbody/body[@name='rig']")
    assert root.find("freejoint") is None and root.find("joint") is None
    if kind == kinematics.FLOATING:
        assert arm.parent is None and arm.joint_parent.name == "rig"
        assert root.find(".//body[@name='arm']") is None
        assert mujoco.find("worldbody/body[@name='arm']/freejoint").get("name") == "arm-base"
    else:
        assert root.find("body[@name='arm']") is not None
    joints = mjcf_joints(mujoco)
    if written is not None:
        assert [(name, kind) for name, kind in joints] == written
    else:
        # A composite is its freedoms in order, one MJCF joint each.
        assert [kind for _name, kind in joints] == [
            "hinge" if freedom.kind == kinematics.TURN else "slide" for freedom in arm.joint.freedoms
        ]
    assert mujoco.find("compiler").get("angle") == "degree"


def test_the_exporter_and_the_simulation_agree_on_the_start_key(export_mjcf):
    import simulate_mujoco

    assert export_mjcf.START_KEY == simulate_mujoco.START_KEY


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


def test_a_screw_is_bounded_by_its_turn_and_a_bound_on_its_move_is_the_turns_through_the_lead(
    tmp_path, stand_in_geometry, export_mjcf
):
    """5 mm either way on a 10 mm thread is half a turn either way, on the hinge; the slide is unbounded."""
    tree = rig(tmp_path, "threadStep: 10\nparameters: {moveZ: [-5, 5]}\nmotion: screw")
    move = kinematics.bodies(tree).by_name("arm").joint.freedoms[1]
    assert (move.lower, move.upper) == (-5.0, 5.0)
    _result, _path, mujoco = export(export_mjcf, tmp_path, tree)

    hinge = mujoco.find(".//joint[@name='arm-base.turnZ']")
    slide = mujoco.find(".//joint[@name='arm-base.moveZ']")
    assert hinge.get("limited") == "true"
    turn = sorted((bound - move.offset) / move.ratio for bound in (move.lower, move.upper))
    assert [float(v) for v in hinge.get("range").split()] == pytest.approx(turn)
    assert turn == pytest.approx([-180.0, 180.0])
    assert slide.get("limited") == "false" and slide.get("range") is None


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


@pytest.mark.parametrize(
    "bore, mjcf_type, finite",
    [
        # 'turnZ' from -30 degrees up, and open above.
        ("motion: {type: revolute, limits: {upper: 30}}", "hinge", -30.0),
        # 'moveZ' up to nought, and open below: metres in the file.
        ("motion: {type: prismatic, limits: {lower: 0}}", "slide", 0.0),
    ],
    ids=["turn", "move"],
)
def test_an_open_side_is_infinite(tmp_path, stand_in_geometry, export_mjcf, bore, mjcf_type, finite):
    """MJCF's range has two ends, and MuJoCo reads 'inf' for one: the limit there measures an infinite distance."""
    tree = rig(tmp_path, bore)
    freedom = kinematics.bodies(tree).by_name("arm").joint.freedoms[0]
    _result, path, mujoco = export(export_mjcf, tmp_path, tree)

    joint = mujoco.find(".//joint[@name='arm-base']")
    assert joint.get("type") == mjcf_type and joint.get("limited") == "true"
    lower, upper = [float(v) for v in joint.get("range").split()]
    if freedom.lower is None:
        assert (lower, upper) == (-math.inf, pytest.approx(finite))
    else:
        assert (lower, upper) == (pytest.approx(finite), math.inf)
    # And a million of anything, which this exporter wrote before, is gone.
    assert "1000000" not in joint.get("range") and "1e+06" not in joint.get("range")

    mujoco_module = pytest.importorskip("mujoco")
    model = mujoco_module.MjModel.from_xml_path(str(path))
    assert bool(model.jnt_limited[0])
    assert sorted(math.isinf(v) for v in model.jnt_range[0]) == [False, True]


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
# Floating: a body of its own
#


def test_a_floating_child_is_a_free_body_with_nothing_excluded_and_read_against_its_base(
    tmp_path, stand_in_geometry, export_mjcf
):
    """A '<freejoint>' in the '<worldbody>', never three slides and a ball under what it is connected to."""
    tree = in_a_scene(rig(tmp_path, "motion: floating"), fixed=True)
    result, path, mujoco = export(export_mjcf, tmp_path, tree, names_file=True)

    arm = mujoco.find("worldbody/body[@name='arm']")
    assert arm is not None and mujoco.find("worldbody/body[@name='subject']//body[@name='arm']") is None
    assert [element.get("name") for element in arm.findall("freejoint")] == ["arm-base"]
    assert arm.find("joint") is None
    # MuJoCo collides it with the body it was connected to, which is what it
    # rests on: no '<exclude>' between them, and none at all here.
    assert mujoco.find("contact") is None
    assert not any("floating" in warning for warning in result["warnings"])
    # Reported as the joint it was declared as, against the body it was
    # connected to.
    entry = naming.read_sidecar(str(path))["joints"]["arm-base"]
    assert entry["path"] == ["arm-base"] and entry["joint"] == "arm-base"
    assert entry["map"]["kind"] == kinematics.FLOATING and entry["map"]["parent"] == "subject"


#
# Balls: re-zeroed, or bounded by a cone about PartCAD's zero
#


def _ball(tmp_path, export_mjcf, bore, start):
    pytest.importorskip("mujoco")
    tree = in_a_scene(rig(tmp_path, bore, params=start), SCENE_LOCATION)
    result, path, xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)
    return kinematics.bodies(tree), result, path, xml


def test_a_ball_with_no_bound_is_written_at_its_start_and_re_zeroed_there(tmp_path, stand_in_geometry, export_mjcf):
    """MuJoCo's identity is the start, and the joint map turns a reading back by it: J' = J . R_s."""
    import mujoco

    found, result, path, xml = _ball(tmp_path, export_mjcf, "motion: ball", ", toParams: {turnX: 10, turnY: 20}")
    joint = found.by_name("arm").joint
    start = kinematics.rezeroed(joint).rotation

    assert xml.find(".//joint[@name='arm-base']").get("limited") == "false"
    assert xml.find("keyframe") is None
    entry = naming.read_sidecar(str(path))["joints"]["arm-base"]["map"]
    assert entry["rotation"] == pytest.approx(list(start))
    # Written where the assembly shows it, at MuJoCo's own start.
    model = mujoco.MjModel.from_xml_path(str(path))
    data = started(mujoco, model)
    assert _close_pose(_body_pose(mujoco, model, data, "arm"), found.by_name("arm").world)
    # And read there as its start: both turns kept, and the third at nought.
    reading = read(mujoco, model, data, path)["arm-base"]
    assert reading["type"] == "ball"
    assert reading["pos"] == pytest.approx({"turnX": 10.0, "turnY": 20.0, "turnZ": 0.0}, abs=1e-9)
    assert _same_rotation(reading["quat"], start)
    assert reading["vel"] == pytest.approx([0.0, 0.0, 0.0])


def test_a_bounded_ball_at_its_zero_is_written_with_its_cone_and_no_key(tmp_path, stand_in_geometry, export_mjcf):
    """'range="0 a"': the one angle a ball turns through, which is how PartCAD bounds it too."""
    import mujoco

    found, result, path, xml = _ball(tmp_path, export_mjcf, "motion: {type: ball, limits: {upper: 20}}", "")
    assert found.by_name("arm").joint.turn_limit == 20.0

    ball = xml.find(".//joint[@name='arm-base']")
    assert (ball.get("limited"), ball.get("range")) == ("true", "0 20")
    assert xml.find("keyframe") is None
    assert not any("bounded" in warning for warning in result["warnings"])
    model = mujoco.MjModel.from_xml_path(str(path))
    assert list(model.jnt_range[0]) == pytest.approx([0.0, math.radians(20.0)])


def test_a_bounded_ball_away_from_its_zero_is_written_at_the_zero_and_started_by_the_key(
    tmp_path, stand_in_geometry, export_mjcf
):
    """Its cone stays about PartCAD's zero: the body at the zero, the start in '<key qpos>', and it is reported."""
    import mujoco

    found, result, path, xml = _ball(
        tmp_path, export_mjcf, "motion: {type: ball, limits: {upper: 20}}", ", toParams: {turnX: 15, turnZ: 30}"
    )
    arm = found.by_name("arm")
    joint = arm.joint
    start = kinematics.rezeroed(joint).rotation

    # Reported, because a file opened by hand shows the ball at its zero.
    (warning,) = [warning for warning in result["warnings"] if "keyframe" in warning]
    assert "'arm-base'" in warning and export_mjcf.START_KEY in warning
    # The cone is written as it is, and the map's start is the identity: the
    # engine's rotation is PartCAD's.
    ball = xml.find(".//joint[@name='arm-base']")
    assert (ball.get("limited"), ball.get("range")) == ("true", "0 20")
    entry = naming.read_sidecar(str(path))["joints"]["arm-base"]["map"]
    assert entry["rotation"] == [1.0, 0.0, 0.0, 0.0]

    model = mujoco.MjModel.from_xml_path(str(path))
    # The body is written at the ball's zero: where the joint puts it with
    # every turn at nought.
    loaded = mujoco.MjData(model)
    mujoco.mj_forward(model, loaded)
    zero = kinematics.compose(arm.joint_parent.world, joint.location([0.0, 0.0, 0.0]))
    assert _close_pose(_body_pose(mujoco, model, loaded, "arm"), zero)
    # The key is the whole of 'qpos' -- MuJoCo's own start for every other
    # coordinate, and this ball's start for its own.
    (key,) = xml.findall("keyframe/key")
    assert key.get("name") == export_mjcf.START_KEY
    assert model.nkey == 1 and len(model.key_qpos[0]) == model.nq
    at = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm-base")]
    for index in range(model.nq):
        if not at <= index < at + 4:
            assert model.key_qpos[0][index] == pytest.approx(model.qpos0[index], abs=1e-12), index
    # Started from it, the body is where the assembly shows it, and reads its start.
    data = started(mujoco, model)
    assert _close_pose(_body_pose(mujoco, model, data, "arm"), arm.world)
    reading = read(mujoco, model, data, path)["arm-base"]
    assert reading["pos"] == pytest.approx({"turnX": 15.0, "turnY": 0.0, "turnZ": 30.0}, abs=1e-9)
    assert _same_rotation(reading["quat"], start)


def test_the_simulation_starts_a_bounded_ball_from_its_key(tmp_path, stand_in_geometry, export_mjcf):
    """'before' is the assembly as it is drawn, not the zero the file had to be written at."""
    import simulate_mujoco

    found, _result, path, _xml = _ball(
        tmp_path, export_mjcf, "motion: {type: ball, limits: {upper: 20}}", ", toParams: {turnX: 15}"
    )
    result = simulate_mujoco.process(str(tmp_path), {"scene_file": str(path), "duration": 0.002})

    before = result["before"]
    assert before["joints"]["arm-base"]["pos"] == pytest.approx({"turnX": 15.0, "turnY": 0.0, "turnZ": 0.0}, abs=1e-9)
    # The ball is centred on the arm's origin, so it is the orientation that
    # says the key was applied: the file holds the arm 15 degrees from this.
    arm = found.by_name("arm")
    assert before["bodies"]["arm"]["pos"] == pytest.approx(list(arm.world[1]), abs=1e-6)
    assert _same_rotation(before["bodies"]["arm"]["quat"], arm.world[0])


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
    """Two poses within 'tolerance': of each quaternion component, and in metres (the poses are in mm)."""
    (qa, ta), (qb, tb) = a, b
    same = max(abs(x - y) for x, y in zip(qa, qb))
    flipped = max(abs(x + y) for x, y in zip(qa, qb))
    return min(same, flipped) <= tolerance and max(abs(x - y) for x, y in zip(ta, tb)) <= tolerance * 1000


def _same_rotation(a, b, tolerance=1e-9):
    return _close_pose((tuple(a), (0.0, 0.0, 0.0)), (tuple(b), (0.0, 0.0, 0.0)), tolerance)


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


@pytest.mark.parametrize("row, bore, kind, written, start", ROWS, ids=ROW_IDS)
def test_every_part_is_where_the_assembly_shows_it_and_a_hinge_or_slide_moves_as_partcad_says(
    tmp_path, stand_in_geometry, export_mjcf, row, bore, kind, written, start
):
    """At the start every part is where the tree puts it; a scalar freedom's 'qpos' is PartCAD's value of it."""
    mujoco = pytest.importorskip("mujoco")
    tree = in_a_scene(rig(tmp_path, bore, params=start), SCENE_LOCATION)
    found = kinematics.bodies(tree)
    _result, path, xml = export(export_mjcf, tmp_path, tree, static=False)

    model = mujoco.MjModel.from_xml_path(str(path))
    data = started(mujoco, model)

    flat = _flat_poses(tree)
    for body in found:
        assert _close_pose(_body_pose(mujoco, model, data, body.name), body.world), body.name
        for index, part in enumerate(body.parts):
            geom = xml.find(".//geom[@name='%s_geom_%d']" % (body.name, index))
            placed = urdf_common.compose(_body_pose(mujoco, model, data, body.name), _geom_pose(geom))
            assert _close_pose(placed, flat[id(part.node)]), (body.name, part.label)

    if kind in (kinematics.BALL, kinematics.FLOATING):
        # A rotation, and a pose: no coordinate of either is a PartCAD value
        # as it stands. Their reading is the next test's.
        return
    # And moved: every freedom away from where it starts, a coupled one by its
    # coupling, set as MuJoCo's own coordinates in radians and metres -- which
    # is PartCAD's value, since every 'ref' is the start -- and the arm is where
    # the joint's own product puts it.
    arm = found.by_name("arm")
    joint = arm.joint
    values = joint.values()
    for freedom in joint.freedoms:
        values[freedom.index] += 17.0 if freedom.kind == kinematics.TURN else 3.0
    for freedom in joint.freedoms:
        if freedom.follows is not None:
            values[freedom.index] = freedom.ratio * values[freedom.follows] + freedom.offset

    names = [name for name, _type in mjcf_joints(xml)]
    for freedom in joint.freedoms:
        name = names[0] if len(names) == 1 else "arm-base." + export_mjcf.freedom_label(joint, freedom)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert index >= 0, name
        scale = math.pi / 180.0 if freedom.kind == kinematics.TURN else 0.001
        data.qpos[model.jnt_qposadr[index]] = values[freedom.index] * scale
    mujoco.mj_forward(model, data)

    expected = kinematics.compose(arm.parent.world, joint.location(values))
    assert _close_pose(_body_pose(mujoco, model, data, "arm"), expected), kind


#
# Readings: MuJoCo's values back as PartCAD's, through the joint map
#


def _moved(mujoco, model, data, index, step):
    """MJCF joint 'index' moved away from where it is, by an amount that differs by 'step'."""
    kind = model.jnt_type[index]
    at = model.jnt_qposadr[index]
    if kind == mujoco.mjtJoint.mjJNT_HINGE:
        data.qpos[at] += 0.3 + 0.1 * step
    elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
        data.qpos[at] += 0.004 + 0.001 * step
    elif kind == mujoco.mjtJoint.mjJNT_BALL:
        turn = urdf_common.axis_angle_to_quat((0.3, -0.5, 0.8), 40.0 + step)
        data.qpos[at : at + 4] = urdf_common.quat_mul(tuple(data.qpos[at : at + 4]), turn)
    else:
        data.qpos[at : at + 3] += [0.012, -0.007, 0.021]
        turn = urdf_common.axis_angle_to_quat((-0.2, 0.9, 0.4), 33.0)
        data.qpos[at + 3 : at + 7] = urdf_common.quat_mul(turn, tuple(data.qpos[at + 3 : at + 7]))


@pytest.mark.parametrize("row, bore, kind, written, start", ROWS, ids=ROW_IDS)
def test_a_reading_is_partcads_value_of_where_mujoco_put_the_body(
    tmp_path, stand_in_geometry, export_mjcf, row, bore, kind, written, start
):
    """Every MJCF joint of the arm moved anywhere, read back as PartCAD's values, is where 'joint.location' says.

    The round trip section 8.8b checked for a hinge, for every kind: what MuJoCo
    makes of its own coordinates is the child's pose against the parent, and
    PartCAD's joint at the values the reading reports puts it there, to 1e-9 m.
    A ball's values are re-zeroed and a floating joint's are a pose, so this is
    what holds 'kinematics.reading()' and the maps the exporter writes to one
    another.
    """
    mujoco = pytest.importorskip("mujoco")
    tree = in_a_scene(rig(tmp_path, bore, params=start), SCENE_LOCATION)
    found = kinematics.bodies(tree)
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)
    arm = found.by_name("arm")
    joint = arm.joint

    model = mujoco.MjModel.from_xml_path(str(path))
    data = started(mujoco, model)
    # At the start it reads the start.
    reading = read(mujoco, model, data, path)["arm-base"]
    assert reading["type"] == kind
    assert values_of(joint, reading) == pytest.approx(joint.values(), abs=1e-9)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm")
    arm_joints = [index for index in range(model.njnt) if model.jnt_bodyid[index] == body]
    assert arm_joints
    for step, index in enumerate(arm_joints):
        _moved(mujoco, model, data, index, step)
    mujoco.mj_forward(model, data)

    reading = read(mujoco, model, data, path)["arm-base"]
    values = values_of(joint, reading)
    parent = _body_pose(mujoco, model, data, arm.joint_parent.name)
    relative = kinematics.compose(kinematics.inverse(parent), _body_pose(mujoco, model, data, "arm"))
    assert _close_pose(relative, joint.location(values)), (kind, values)
    if kind == kinematics.BALL:
        # Its rotation, in its own frame from PartCAD's zero.
        classification = joint.classification
        turned = kinematics.compose(
            kinematics.compose(classification.origin, (tuple(reading["quat"]), (0.0, 0.0, 0.0))),
            classification.tail,
        )
        assert _close_pose(relative, turned)
    if kind == kinematics.FLOATING:
        classification = joint.classification
        local = kinematics.compose(
            kinematics.compose(kinematics.inverse(classification.origin), relative),
            kinematics.inverse(classification.tail),
        )
        assert _same_rotation(reading["quat"], local[0])


def test_a_joint_of_several_freedoms_is_read_as_one_under_partcads_name(tmp_path, stand_in_geometry, export_mjcf):
    """Not '<name>.<parameter>' any more: one entry, its 'pos', 'vel' and 'effort' keyed by parameter."""
    mujoco = pytest.importorskip("mujoco")
    import simulate_mujoco

    tree = in_a_scene(rig(tmp_path, "threadStep: 10\nmotion: screw", params=", toParams: {turnZ: 36}"))
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)
    result = simulate_mujoco.process(str(tmp_path), {"scene_file": str(path), "duration": 0.01})

    for reading in (result["before"], result["after"]):
        assert list(reading["joints"]) == ["arm-base"]
        nut = reading["joints"]["arm-base"]
        assert nut["type"] == kinematics.SCREW
        assert set(nut["pos"]) == set(nut["vel"]) == set(nut["effort"]) == {"turnZ", "moveZ"}
    assert result["before"]["joints"]["arm-base"]["pos"]["turnZ"] == pytest.approx(36.0, abs=1e-9)
    assert result["before"]["joints"]["arm-base"]["effort"] == {"turnZ": 0.0, "moveZ": 0.0}
    # Two MJCF joints, one reading.
    assert mujoco.MjModel.from_xml_path(str(path)).njnt == 2


def test_a_floating_joint_moves_at_the_rate_its_reading_changes_at(tmp_path, stand_in_geometry, export_mjcf):
    """Falling and spinning against a base that falls and spins too, each of its six 'vel' is d'pos'/dt.

    And its 'angular' is how fast its 'quat' turns, about its moving frame.
    The base is free as well, so that what is checked is the block's motion
    *against* it -- the two bodies' velocities, from 'mj_objectVelocity', put
    together by PartCAD's 'reading()' -- and not the block's in the world. Read
    every step of a run small enough for a central difference to be exact to
    many digits, a tenth of a millisecond under RK4, and short enough that
    nothing touches anything.
    """
    mujoco = pytest.importorskip("mujoco")
    tree = in_a_scene(
        rig(tmp_path, "motion: floating", params=", toParams: {moveZ: -100}"),
        [[0.0, 0.0, 500.0], [0.0, 0.0, 1.0], 0.0],
        fixed=False,
    )
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True, ground_plane=False)
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.timestep = step = 1e-4
    data = started(mujoco, model)
    # A free joint's velocity in MuJoCo is its frame's linear velocity in the
    # world, m/s, then its angular velocity about its own axes, rad/s.
    for body, twist in (("subject", [0.1, 0.2, 0.3, 1.0, -0.5, 0.7]), ("arm", [-0.2, 0.1, 0.4, 3.0, -2.0, 5.0])):
        joint = model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]
        assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        data.qvel[model.jnt_dofadr[joint] : model.jnt_dofadr[joint] + 6] = twist
    mujoco.mj_forward(model, data)

    readings = []
    for _ in range(200):
        readings.append(read(mujoco, model, data, path)["arm-base"])
        mujoco.mj_step(model, data)
    assert data.ncon == 0

    def spun(earlier, now, later):
        """The angular velocity a quaternion's change says, deg/s about its own axes: 2 q^-1 dq/dt."""
        q = now
        # The same rotation is q and -q: each neighbour taken on q's side.
        a, b = [(-1.0 if sum(x * y for x, y in zip(other, q)) < 0 else 1.0) for other in (earlier, later)]
        rate = [(b * y - a * x) / (2.0 * step) for x, y in zip(earlier, later)]
        product = urdf_common.quat_mul((q[0], -q[1], -q[2], -q[3]), tuple(rate))
        return [2.0 * math.degrees(v) for v in product[1:]]

    names = kinematics.bodies(tree).by_name("arm").joint.parameter_names()
    for earlier, now, later in zip(readings, readings[1:], readings[2:]):
        assert set(now["vel"]) == set(names) | {"angular"}
        for name in names:
            changed = (later["pos"][name] - earlier["pos"][name]) / (2.0 * step)
            assert now["vel"][name] == pytest.approx(changed, rel=1e-5, abs=1e-6), name
        assert now["vel"]["angular"] == pytest.approx(spun(earlier["quat"], now["quat"], later["quat"]), rel=1e-5)
    # And it moved, every freedom of it: a check of rates of nought would pass.
    final = readings[-1]["vel"]
    assert all(abs(final[name]) > 1.0 for name in names), final


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


def paddle():
    """A paddle 300 mm long and 150 mm wide, 200 g, along the arm's X from the joint, flat in its X-Y plane.

    What a ball's cone is checked with rather than the bar: MuJoCo softens a
    ball's limit by the inverse inertia of all three of its freedoms, which it
    averages, so a body that barely resists turning about its own length -- a
    thin bar -- is barely stopped by its cone at all. The bar above, on a ball
    bounded at 20 degrees, swings to 168; a real 300 x 20 x 10 mm bar to 45.
    """
    width = 0.15
    return {
        "volume": 1000.0,
        "mass": BAR_MASS,
        "centerOfMass": [BAR_D * 1000.0, 0.0, 0.0],
        "inertia": {
            "ixx": BAR_MASS * width**2 / 12.0,
            "iyy": BAR_MASS * BAR_LENGTH**2 / 12.0,
            "izz": BAR_MASS * (BAR_LENGTH**2 + width**2) / 12.0,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        },
    }


def pendulum(
    tmp_path, export_mjcf, start=0.0, duration=1.0, samples=1000, bore="motion: revolute", params=None, physics=None
):
    """The bar released at 'start' degrees from horizontal, about a horizontal axis.

    The joint's axis is the bore's Z axis, which the scene turns to lie along
    world Y, so that the bar, along the arm's X, starts horizontal at nought.
    Exported as the simulation asks for it, with the sidecar a run reads
    PartCAD's names out of.
    """
    import simulate_mujoco

    if params is None:
        params = ", toParams: {turnZ: %s}" % start if start else ""
    tree = in_a_scene(rig(tmp_path, bore, params=params), [[0.0, 0.0, 500.0], [1.0, 0.0, 0.0], 90.0])
    _result, path, _xml = export(
        export_mjcf,
        tmp_path,
        tree,
        "pendulum.xml",
        static=False,
        names_file=True,
        properties=weighed(tree, {"arm": physics or bar()}),
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
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)
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
    """Pulled down its axis by its weight, the nut turns 360 degrees for every 10 mm it moves.

    Read as one joint, under PartCAD's name for it, with a value per parameter.
    """
    pytest.importorskip("mujoco")
    import simulate_mujoco

    tree = in_a_scene(rig(tmp_path, "threadStep: 10\nmotion: screw"), [[0.0, 0.0, 800.0], [0.0, 0.0, 1.0], 0.0])
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)
    result = simulate_mujoco.process(
        str(tmp_path), {"scene_file": str(path), "duration": 0.5, "samples": 20, "timestep": 0.0005}
    )

    move = kinematics.bodies(tree).by_name("arm").joint.freedoms[1]
    after = result["after"]["joints"]
    assert after["arm-base"]["type"] == kinematics.SCREW
    moved = after["arm-base"]["pos"]["moveZ"]
    assert abs(moved) > 20.0
    for reading in [sample["joints"] for sample in result["samples"]] + [after]:
        nut = reading["arm-base"]
        assert nut["pos"]["moveZ"] == pytest.approx(move.ratio * nut["pos"]["turnZ"] + move.offset, abs=0.05)
        assert nut["vel"]["moveZ"] == pytest.approx(move.ratio * nut["vel"]["turnZ"], abs=0.5)


def test_a_loose_block_falls_as_anything_falls_and_comes_to_rest_on_its_base(tmp_path, stand_in_geometry, export_mjcf):
    """'floating': released 30 mm above where it sits on the base, it falls 1/2 g t^2 and lands *on* it.

    Written as three slides and a ball under the base, as it once was, it
    would have been excluded from colliding with the base, and fallen through.
    Thirty millimetres and not more because the block is a 10 mm cube, and
    MuJoCo's contacts are soft: dropped from 100 mm it sinks most of its own
    height into the base before it is stopped, and slides off.
    """
    pytest.importorskip("mujoco")
    import simulate_mujoco

    # 'moveZ' runs down here (the pin is turned over to face the bore), so
    # -30 is 30 mm up: the block's underside 30 mm above the base's top.
    tree = in_a_scene(
        rig(tmp_path, "motion: floating", params=", toParams: {moveZ: -30}"), [[0.0, 0.0, 500.0], [0.0, 0.0, 1.0], 0.0]
    )
    found = kinematics.bodies(tree)
    top = found.by_name("subject").world[1][2] + 10.0
    assert found.by_name("arm").world[1][2] == pytest.approx(top + 30.0)
    _result, path, _xml = export(export_mjcf, tmp_path, tree, static=False, names_file=True)

    # 50 ms: 12.3 mm down, and still 17.7 mm in the air.
    falling = simulate_mujoco.process(str(tmp_path), {"scene_file": str(path), "duration": 0.05, "timestep": 0.0005})
    landed = simulate_mujoco.process(str(tmp_path), {"scene_file": str(path), "duration": 1.0, "timestep": 0.0005})

    t = falling["after"]["time"]
    fallen = falling["before"]["bodies"]["arm"]["pos"][2] - falling["after"]["bodies"]["arm"]["pos"][2]
    assert fallen == pytest.approx(0.5 * G * t * t * 1000.0, rel=0.03)
    # The joint reads the same fall, against the base -- 'moveZ' runs down --
    # and only that: the block does not turn.
    before, after = falling["before"]["joints"]["arm-base"], falling["after"]["joints"]["arm-base"]
    assert after["type"] == kinematics.FLOATING
    assert before["pos"]["moveZ"] == pytest.approx(-30.0, abs=1e-9)
    assert after["pos"]["moveZ"] - before["pos"]["moveZ"] == pytest.approx(fallen, abs=0.5)
    assert after["quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)
    # And it stops on the base: its underside on the base's top, give or take
    # how far MuJoCo's soft contact lets one into the other.
    assert landed["after"]["bodies"]["arm"]["pos"][2] == pytest.approx(top, abs=0.5)
    assert landed["after"]["joints"]["arm-base"]["pos"]["moveZ"] == pytest.approx(0.0, abs=0.5)


def _tilt(quat):
    """How far a rotation turns, in degrees: the angle a ball's cone bounds."""
    return math.degrees(2.0 * math.acos(min(1.0, abs(quat[0]))))


CONE = "motion: {type: ball, limits: {upper: 20}}"


def test_a_ball_bounded_by_a_cone_stops_at_it(tmp_path, stand_in_geometry, export_mjcf):
    """Released horizontal on a ball bounded at 20 degrees, the paddle falls to 20 and stays there.

    Unbounded, the same paddle swings right over, as a pendulum released
    horizontal does.
    """
    pytest.importorskip("mujoco")

    bounded = pendulum(tmp_path / "cone", export_mjcf, samples=200, bore=CONE, physics=paddle())
    free = pendulum(tmp_path / "ball", export_mjcf, samples=200, bore="motion: ball", physics=paddle())

    tilts = [_tilt(sample["joints"]["arm-base"]["quat"]) for sample in bounded["samples"]]
    assert _tilt(bounded["before"]["joints"]["arm-base"]["quat"]) == pytest.approx(0.0, abs=1e-9)
    # A stop's own give, which is MuJoCo's softness and not the joint's.
    assert max(tilts) < 25.0
    assert tilts[-1] == pytest.approx(20.0, abs=1.5)
    assert max(_tilt(sample["joints"]["arm-base"]["quat"]) for sample in free["samples"]) > 170.0


@pytest.mark.parametrize("start", [15.0, -15.0], ids=["below", "above"])
def test_a_bounded_ball_started_away_from_its_zero_keeps_its_cone_about_the_zero(
    tmp_path, stand_in_geometry, export_mjcf, start
):
    """It comes to rest against the cone 20 degrees from PartCAD's zero, wherever it starts.

    Started 15 degrees from the horizontal, one way or the other: written
    re-zeroed at its start, the cone would be about the start instead, and
    hold it 35 degrees from the horizontal, or 5.
    """
    pytest.importorskip("mujoco")

    result = pendulum(
        tmp_path, export_mjcf, samples=200, bore=CONE, params=", toParams: {turnZ: %s}" % start, physics=paddle()
    )

    assert result["before"]["joints"]["arm-base"]["pos"]["turnZ"] == pytest.approx(start, abs=1e-9)
    tilts = [_tilt(sample["joints"]["arm-base"]["quat"]) for sample in result["samples"]]
    assert max(tilts) < 25.0
    assert tilts[-1] == pytest.approx(20.0, abs=1.5)
