#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""What a run says acts on each body (see 'body_loads' in 'simulate_mujoco.py').

The models are written by hand, as in 'test_joints.py', so that what is under
test is the reading: a box resting on the ground, an arm held level by the
spring of its hinge from a frame welded to the world, the same arm swinging
free, and a float held under water by its hinge. Each is checked against what
statics or Newton says it has to be -- the ground holds the box up with its
weight, the hinge holds the arm with its weight and the moment of it, the weld
holds both -- and every body against the one rule a set of loads has to keep:
they add up to the body's mass times its acceleration. None of it is true in the
wrong units or with a sign the wrong way round, which is the other half of the
point.

MuJoCo is needed for all of it, and the whole file is skipped without one; CI
installs it.
"""

import math
import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")

import simulate_mujoco  # noqa: E402

G = 9.81
BOX = 2.0
# The arm: a 1 kg ball 0.5 m out along +x from a hinge about +y, on a frame of
# 3 kg welded to the world. Its spring is stiff enough to hold it within a few
# degrees of level, and damped so that it comes to rest there.
ARM, REACH, FRAME = 1.0, 0.5, 3.0
STIFFNESS = 400.0

MODEL = """
<mujoco model="loads">
  <option gravity="0 0 -%(g)s" timestep="0.001"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="%(box)s"/>
    </body>
    <body name="frame" pos="1 0 1">
      <geom type="box" size="0.02 0.02 0.02" mass="%(frame)s" contype="0" conaffinity="0"/>
      <body name="arm">
        <joint name="hold" type="hinge" axis="0 1 0" stiffness="%(stiffness)s" damping="20"/>
        <geom type="sphere" size="0.01" pos="%(reach)s 0 0" mass="%(arm)s" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="pendulum" pos="-1 0 1">
      <joint name="swing" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.01" pos="%(reach)s 0 0" mass="%(arm)s" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""" % {
    "g": G,
    "box": BOX,
    "frame": FRAME,
    "arm": ARM,
    "reach": REACH,
    "stiffness": STIFFNESS,
}


def run(tmp_path, scene=MODEL, **request):
    (tmp_path / "scene.xml").write_text(scene, encoding="utf-8")
    request.setdefault("scene_file", str(tmp_path / "scene.xml"))
    request.setdefault("duration", 3.0)
    result = simulate_mujoco.process(str(tmp_path), request)
    assert result["success"] is True
    return result


def loads_of(reading, body, kind=None, source=None):
    found = reading["bodies"][body]["loads"]
    return [
        load for load in found if (kind is None or load["kind"] == kind) and (source is None or load["from"] == source)
    ]


def total(loads, key="force"):
    return [sum(load[key][axis] for load in loads) for axis in range(3)]


def moment_about(loads, point):
    """The moment of every load about 'point' (mm), N*m."""
    moment = [0.0, 0.0, 0.0]
    for load in loads:
        r = [(load["at"][axis] - point[axis]) / 1000.0 for axis in range(3)]
        f = load["force"]
        cross = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
        moment = [moment[axis] + load["torque"][axis] + cross[axis] for axis in range(3)]
    return moment


def test_the_ground_holds_a_box_up_with_its_weight(tmp_path):
    after = run(tmp_path)["after"]
    box = after["bodies"]["box"]
    assert box["mass"] == pytest.approx(BOX)
    (ground,) = loads_of(after, "box", "contact", "world")
    # Four corners on the plane, pushing up, as hard as the box weighs: newtons,
    # on the box, in the world's frame.
    assert ground["contacts"] == 4
    assert ground["force"] == pytest.approx([0.0, 0.0, BOX * G], abs=1e-3)
    # At the middle of the face it stands on, in millimetres.
    assert ground["at"] == pytest.approx([0.0, 0.0, 0.0], abs=0.5)
    (weight,) = loads_of(after, "box", "gravity")
    assert weight["force"] == pytest.approx([0.0, 0.0, -BOX * G])
    assert weight["at"] == pytest.approx(box["com"])
    # A free body's six freedoms pass nothing: nothing else acts on it.
    assert len(box["loads"]) == 2
    assert box["acc"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_a_hinge_holds_an_arm_with_its_weight_and_the_moment_of_it(tmp_path):
    after = run(tmp_path)["after"]
    arm = after["bodies"]["arm"]
    (held,) = loads_of(after, "arm", "joint", "frame")
    assert held["joint"] == "hold"
    # At the hinge, which does not move: the frame's origin, in millimetres.
    assert held["at"] == pytest.approx([1000.0, 0.0, 1000.0], abs=1e-6)
    assert held["force"] == pytest.approx([0.0, 0.0, ARM * G], abs=1e-3)
    # The spring holds the moment the arm's weight has about the hinge, about
    # +y: the arm is out along +x and pulled down along -z, so its weight turns
    # it about +y and the hinge holds it the other way. Its lever is where the
    # arm came to rest, a few degrees below level.
    lever = (arm["com"][0] - 1000.0) / 1000.0
    assert 0.49 < lever <= REACH
    assert held["torque"] == pytest.approx([0.0, -ARM * G * lever, 0.0], abs=1e-3)
    # Which is the spring's own: its stiffness times how far it has turned.
    angle = math.radians(after["joints"]["hold"]["pos"])
    assert abs(held["torque"][1]) == pytest.approx(STIFFNESS * abs(angle), rel=1e-3)


def test_the_body_above_feels_the_joint_the_other_way_and_its_weld_holds_everything(tmp_path):
    after = run(tmp_path)["after"]
    (from_arm,) = loads_of(after, "frame", "joint", "arm")
    (on_arm,) = loads_of(after, "arm", "joint", "frame")
    assert from_arm["joint"] == "hold"
    assert from_arm["at"] == pytest.approx(on_arm["at"])
    assert from_arm["force"] == pytest.approx([-v for v in on_arm["force"]])
    assert from_arm["torque"] == pytest.approx([-v for v in on_arm["torque"]])
    # The frame is welded to the world, which holds up the frame and the arm,
    # and does not move at all.
    assert after["bodies"]["frame"]["acc"] == [0.0, 0.0, 0.0]
    assert after["bodies"]["frame"]["angvel"] == [0.0, 0.0, 0.0]
    (weld,) = loads_of(after, "frame", "fixed", "world")
    assert weld["at"] == pytest.approx([1000.0, 0.0, 1000.0])
    assert weld["force"] == pytest.approx([0.0, 0.0, (FRAME + ARM) * G], abs=1e-3)
    assert moment_about(after["bodies"]["frame"]["loads"], weld["at"]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_the_loads_on_a_moving_body_add_up_to_its_mass_times_its_acceleration(tmp_path):
    result = run(tmp_path, duration=0.3, samples=5)
    for reading in result["samples"] + [result["after"]]:
        pendulum = reading["bodies"]["pendulum"]
        (held,) = loads_of(reading, "pendulum", "joint", "world")
        assert held["joint"] == "swing"
        # A hinge with nothing on it holds no moment about its own axis.
        assert held["torque"][1] == pytest.approx(0.0, abs=1e-6)
        expected = [pendulum["mass"] * a for a in pendulum["acc"]]
        assert total(pendulum["loads"]) == pytest.approx(expected, abs=1e-3)
    # Falling from level, it is moving and turning faster all the way down.
    pendulum = result["after"]["bodies"]["pendulum"]
    assert abs(pendulum["angvel"][1]) > 90
    assert abs(sum(a * a for a in pendulum["acc"])) > 1.0


def test_a_float_on_a_hinge_is_held_by_the_lift_and_the_hinge_carries_the_rest(tmp_path):
    # Half its weight lifted, at a centre of buoyancy 10 mm above its centre of
    # mass, as the exporter writes a body in a fluid.
    scene = """
    <mujoco model="float">
      <option gravity="0 0 -9.81" timestep="0.001"/>
      <custom><numeric name="partcad:centre_of_buoyancy:float" data="0.5 0 0.01"/></custom>
      <worldbody>
        <body name="float" pos="0 0 1" gravcomp="0.5">
          <joint name="pin" type="hinge" axis="0 1 0" stiffness="400" damping="20"/>
          <geom type="sphere" size="0.01" pos="0.5 0 0" mass="1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    after = run(tmp_path, scene)["after"]
    (lift,) = loads_of(after, "float", "buoyancy")
    assert lift["force"] == pytest.approx([0.0, 0.0, 0.5 * G], abs=1e-6)
    com = after["bodies"]["float"]["com"]
    assert lift["at"][2] == pytest.approx(com[2] + 10.0, abs=0.5)
    (held,) = loads_of(after, "float", "joint", "world")
    assert held["force"] == pytest.approx([0.0, 0.0, 0.5 * G], abs=1e-3)
    assert total(after["bodies"]["float"]["loads"]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_a_run_that_asks_for_no_loads_reports_poses_alone(tmp_path):
    result = run(tmp_path, duration=0.05, loads=False)
    for reading in (result["before"], result["after"]):
        assert sorted(reading["bodies"]["box"]) == ["pos", "quat"]
    assert "parts" not in result


def test_before_is_read_with_its_loads_too(tmp_path):
    before = run(tmp_path, duration=0.05)["before"]
    assert before["time"] == 0.0
    # Released level and at rest, the pendulum starts by falling freely: its
    # centre accelerates straight down at g, so the hinge holds nothing yet --
    # the instant every pendulum starts from.
    pendulum = before["bodies"]["pendulum"]
    assert pendulum["acc"] == pytest.approx([0.0, 0.0, -G], rel=1e-3, abs=1e-3)
    (held,) = loads_of(before, "pendulum", "joint", "world")
    assert held["force"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-2)
    assert pendulum["angvel"] == pytest.approx([0.0, 0.0, 0.0])


def test_a_run_says_which_body_each_part_is_in(tmp_path):
    import partcad as pc

    sys.path.append(os.path.join(os.path.dirname(pc.__file__), "wrappers"))
    import naming

    if not hasattr(naming, "parts_of"):
        pytest.skip("this PartCAD cannot say which parts a body is (partcad/partcad#823)")
    (tmp_path / "scene.xml").write_text(MODEL, encoding="utf-8")
    naming.write_sidecar(
        str(tmp_path / "scene.xml"),
        "mjcf",
        {"body": {}},
        {
            "box": {"path": ["box"], "parts": [{"label": "box", "part": "//p:box", "placement": None}]},
            "arm": {
                "path": ["arm"],
                "parts": [{"label": "rig/arm", "part": "//p:arm", "placement": [[500, 0, 0], [0, 0, 1], 0]}],
            },
        },
    )
    result = run(tmp_path, duration=0.05)
    assert result["parts"] == {
        "box": {"part": "//p:box", "body": "box", "placement": None},
        "rig/arm": {"part": "//p:arm", "body": "arm", "placement": [[500, 0, 0], [0, 0, 1], 0]},
    }
