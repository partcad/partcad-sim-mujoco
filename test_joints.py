#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""What a run says about the joints of a model (see 'joints' in 'simulate_mujoco.py').

PartCAD's own exporter writes every body free today, so the models here are
written by hand: a pendulum on a hinge and the same pendulum on a ball joint, a
carriage falling along a slide and one an actuator holds up, a limited hinge,
an unnamed one, and a free body.

What is checked is physics with a known answer rather than whatever the
plugin happened to print the day the test was written. A pendulum released
from horizontal passes the bottom at the speed its own fall gives it, and
reaches the bottom a quarter period later -- a period that, at this amplitude,
is the complete elliptic integral and not the small-angle formula. A carriage on
a slide falls as anything falls. None of that is true in the wrong units, which
is the other half of the point: degrees rather than radians, millimetres
rather than metres, N rather than anything else.

MuJoCo is needed for all of it, and the whole file is skipped without one; CI
installs it.
"""

import math

import pytest

mujoco = pytest.importorskip("mujoco")

import simulate_mujoco  # noqa: E402

G = 9.81
# The pendulum: a 1 kg ball of 10 mm radius, half a metre out along +x from a
# hinge about +y, so turning the hinge the positive way swings it down.
MASS, ARM, RADIUS = 1.0, 0.5, 0.01
# Its moment of inertia about the hinge: the ball's own, carried out to the arm.
INERTIA = MASS * ARM**2 + 0.4 * MASS * RADIUS**2
# The carriage the actuator holds up, and so the force it takes to.
CARRIAGE = 2.0

JOINTS = """
<mujoco model="joints">
  <!-- RK4 because what is checked is that energy is conserved, and the default
       semi-implicit Euler leaks a little of it every step. -->
  <option gravity="0 0 -%(g)s" timestep="0.001" integrator="RK4"/>
  <!-- Nothing touches anything: every body here moves only as its joint lets
       gravity move it. -->
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="swing" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="%(radius)s" pos="%(arm)s 0 0" mass="%(mass)s"/>
    </body>
    <body name="bob" pos="0 2 1">
      <joint name="socket" type="ball"/>
      <geom type="sphere" size="%(radius)s" pos="%(arm)s 0 0" mass="%(mass)s"/>
    </body>
    <body name="carriage" pos="0 4 1">
      <joint name="drop" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.01 0.01 0.01" mass="%(carriage)s"/>
    </body>
    <body name="held" pos="0 6 1">
      <joint name="hold" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.01 0.01 0.01" mass="%(carriage)s"/>
    </body>
    <body name="flap" pos="0 8 1">
      <joint name="flap" type="hinge" axis="1 0 0" range="-45 45"/>
      <geom type="box" size="0.01 0.01 0.01" mass="1"/>
    </body>
    <body name="wheel" pos="0 10 1">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.01 0.01 0.01" mass="1"/>
    </body>
    <body name="loose" pos="0 12 1">
      <freejoint/>
      <geom type="box" size="0.01 0.01 0.01" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <!-- A constant force and nothing else (no gain, an affine bias), through a
         gear of two: half the carriage's weight, doubled on the way out. -->
    <general name="holder" joint="hold" gear="2" gainprm="0" biastype="affine" biasprm="%(half_weight)s"/>
  </actuator>
</mujoco>
""" % {
    "g": G,
    "mass": MASS,
    "arm": ARM,
    "radius": RADIUS,
    "carriage": CARRIAGE,
    "half_weight": CARRIAGE * G / 2,
}


def complete_elliptic_integral(k):
    """K(k), by the arithmetic-geometric mean: K = pi / (2 AGM(1, sqrt(1 - k^2)))."""
    a, b = 1.0, math.sqrt(1.0 - k * k)
    while abs(a - b) > 1e-15:
        a, b = (a + b) / 2.0, math.sqrt(a * b)
    return math.pi / (2.0 * a)


# A quarter of the pendulum's period at an amplitude of 90 degrees, which is
# how long it takes to fall from horizontal to the bottom.
QUARTER = math.sqrt(INERTIA / (MASS * G * ARM)) * complete_elliptic_integral(math.sin(math.radians(45)))


def run(tmp_path, scene=JOINTS, **request):
    (tmp_path / "scene.xml").write_text(scene, encoding="utf-8")
    request.setdefault("scene_file", str(tmp_path / "scene.xml"))
    request.setdefault("duration", QUARTER)
    result = simulate_mujoco.process(str(tmp_path), request)
    assert result["success"] is True
    return result


@pytest.fixture
def swung(tmp_path):
    """The model run for exactly the time the pendulum takes to reach the bottom."""
    return run(tmp_path, samples=40)


def test_a_pendulum_released_from_horizontal_passes_the_bottom_at_the_speed_its_fall_gives_it(swung):
    """Half the moment of inertia times the square of the speed is what it lost
    in height. In degrees per second, which is the unit an interface's
    'maxVelocity' is in."""
    swing = swung["after"]["joints"]["swing"]

    assert swing["pos"] == pytest.approx(90.0, abs=1.0)
    bottom = math.degrees(math.sqrt(2 * MASS * G * ARM / INERTIA))
    assert swing["vel"] == pytest.approx(bottom, rel=1e-3)


def test_on_the_way_down_the_pendulum_loses_no_energy(swung):
    """At every instant, not only at the bottom: a reading whose angle and speed
    came from two different instants would not pass this."""
    for reading in [swung["before"]] + swung["samples"] + [swung["after"]]:
        swing = reading["joints"]["swing"]
        fallen = MASS * G * ARM * math.sin(math.radians(swing["pos"]))
        moving = 0.5 * INERTIA * math.radians(swing["vel"]) ** 2
        assert moving == pytest.approx(fallen, abs=1e-4)


def test_the_joints_are_read_at_the_time_the_reading_states(swung):
    """Exactly, not a step before it. RK4 integrates a constant acceleration
    without error, so a carriage falling along a slide is where half g t
    squared says at the 'time' beside it, to the last digit -- and moving at
    g t, which is the case that needs 'joints' to evaluate the state afresh
    rather than read what the step left behind."""
    for reading in swung["samples"] + [swung["after"]]:
        t = reading["time"]
        drop = reading["joints"]["drop"]
        assert drop["pos"] == pytest.approx(-0.5 * G * t * t * 1000.0, rel=1e-9)
        assert drop["vel"] == pytest.approx(-G * t * 1000.0, rel=1e-9)


def test_before_anything_moves_the_joints_and_the_bodies_say_the_same_thing(swung):
    """The hinge's angle is the angle its body has turned through. After a step
    the bodies are where MuJoCo last evaluated them, up to a step behind
    'time', so this holds exactly only where nothing has stepped yet."""
    w, _x, y, _z = swung["before"]["bodies"]["arm"]["quat"]
    assert swung["before"]["joints"]["swing"]["pos"] == pytest.approx(math.degrees(2 * math.atan2(y, w)), abs=1e-9)
    assert swung["before"]["bodies"]["carriage"]["pos"][2] == pytest.approx(
        1000.0 + swung["before"]["joints"]["drop"]["pos"], abs=1e-9
    )


def test_the_same_pendulum_on_a_ball_joint_turns_the_same_way_and_says_so_as_a_quaternion(swung):
    """Nothing pushes it out of the plane it starts in, so it is the hinge's
    swing told the other way: a turn about +y, and an angular velocity about the
    body's own y axis that is the hinge's speed."""
    swing = swung["after"]["joints"]["swing"]
    socket = swung["after"]["joints"]["socket"]

    w, x, y, z = socket["quat"]
    assert w * w + x * x + y * y + z * z == pytest.approx(1.0, abs=1e-9)
    assert math.degrees(2 * math.atan2(y, w)) == pytest.approx(swing["pos"], abs=1e-3)
    assert socket["vel"] == pytest.approx([0.0, swing["vel"], 0.0], abs=1e-2)
    assert socket["effort"] == [0.0, 0.0, 0.0]


def test_the_effort_is_what_the_actuator_exerts_along_the_joint_in_newtons(swung):
    """Its weight, exactly, through the gear -- and so the carriage does not move."""
    hold = swung["after"]["joints"]["hold"]

    assert hold["effort"] == pytest.approx(CARRIAGE * G)
    assert hold["pos"] == pytest.approx(0.0, abs=1e-6)
    assert hold["vel"] == pytest.approx(0.0, abs=1e-6)
    # And nothing drives the others.
    assert swung["after"]["joints"]["swing"]["effort"] == 0.0


def test_every_joint_is_named_typed_and_reported_but_a_free_one(swung):
    """A hinge is 'revolute' when it is limited and 'continuous' when it is not,
    the line PartCAD's 'motion:' draws. A free joint is the body's own pose,
    which 'bodies' states already. An unnamed joint is named after its index."""
    reported = swung["before"]["joints"]
    unnamed = mujoco.MjModel.from_xml_string(JOINTS).njnt - 2  # the wheel's: the last but the free one

    assert {name: joint["type"] for name, joint in reported.items()} == {
        "swing": "continuous",
        "socket": "ball",
        "drop": "prismatic",
        "hold": "prismatic",
        "flap": "revolute",
        "joint_%d" % unnamed: "continuous",
    }
    assert "loose" in swung["before"]["bodies"]


def test_before_anything_moves_every_joint_is_where_the_model_put_it(swung):
    for name, joint in swung["before"]["joints"].items():
        if joint["type"] == "ball":
            assert joint["quat"] == [1.0, 0.0, 0.0, 0.0], name
            assert joint["vel"] == [0.0, 0.0, 0.0], name
        else:
            assert joint["pos"] == 0.0, name
            assert joint["vel"] == 0.0, name


def test_a_joint_the_model_starts_away_from_zero_reads_where_its_ref_puts_it(tmp_path):
    """'pos' is the joint's own coordinate, in degrees whatever the file was
    written in: a hinge whose 'ref' is a sixth of a turn reads 60."""
    scene = JOINTS.replace('<mujoco model="joints">', '<mujoco model="joints">\n  <compiler angle="radian"/>')
    scene = scene.replace(
        'type="hinge" axis="1 0 0" range="-45 45"', 'type="hinge" axis="1 0 0" ref="%r"' % (math.pi / 3)
    )

    flap = run(tmp_path, scene, duration=0.01)["before"]["joints"]["flap"]

    assert flap["pos"] == pytest.approx(60.0)
    assert flap["type"] == "continuous"


def test_the_samples_carry_the_joints_too(swung):
    assert swung["samples"]
    assert all(set(sample["joints"]) == set(swung["after"]["joints"]) for sample in swung["samples"])


def test_a_model_without_joints_says_so_with_an_empty_object(tmp_path):
    """Rather than leaving the key out, so that a validation can walk
    'after["joints"]' without asking first whether there is one. Free bodies
    are all a scene PartCAD exports has today, and none of them is a joint here."""
    import test_snapshot

    result = run(tmp_path, test_snapshot.FALLING, duration=0.5, samples=2)

    assert result["before"]["joints"] == {}
    assert result["after"]["joints"] == {}
    assert all(sample["joints"] == {} for sample in result["samples"])
    assert set(result["after"]["bodies"]) == {"bottom", "top"}
