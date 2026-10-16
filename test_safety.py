#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""What stops a run that cannot be trusted, or will not end (see 'simulate_mujoco.py').

Each net is proven by a model, or a request, that needs it:

* a model that diverges, which MuJoCo resets to where it started and goes on
  running -- so that without the check the run ends "unmoved" and passes a
  validation that nothing fell over;
* a run that will not finish within the wall-clock budget PartCAD hands it,
  which stops itself and says how far it got;
* a request for an absurd run, which is refused before anything is stepped.

Every test takes a fraction of a second, or a second where a budget is spent.
MuJoCo is needed for the runs, and they are skipped without one; CI installs it.
"""

import math
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulate_mujoco  # noqa: E402

# A pendulum on a spring far too stiff for its mass and its step, wound half a
# radian from rest ('springref'): the first step's acceleration is past anything
# MuJoCo will integrate.
DIVERGES = """
<mujoco model="diverges">
  <option timestep="0.01"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="swing" type="hinge" axis="0 1 0" stiffness="1e12" springref="0.5"/>
      <geom type="sphere" size="0.01" pos="0.5 0 0" mass="0.001" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# One free block over a floor, which is a well-behaved model that takes as long
# as it is asked to.
FALLS = """
<mujoco model="falls">
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="block" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path, text, name="scene.xml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _diverging(tmp_path):
    return _write(tmp_path, DIVERGES)


# --------------------------------------------------------------------------- #
# A model that diverges                                                       #
# --------------------------------------------------------------------------- #


def test_a_model_that_diverges_stops_the_run_naming_when_and_where(tmp_path):
    pytest.importorskip("mujoco")
    scene = _diverging(tmp_path)

    with pytest.raises(simulate_mujoco.Diverged) as raised:
        simulate_mujoco.process(str(tmp_path), {"scene_file": scene, "duration": 1.0})

    message = str(raised.value)
    assert "t = 0 s" in message
    assert "joint 'swing'" in message and "body 'arm'" in message
    assert "mjWARN_BADQACC" in message
    assert "timestep" in message  # and what to try
    assert "MuJoCo said: Nan, Inf or huge value in QACC" in message  # in its own words


def test_what_mujoco_says_is_kept_rather_than_written_beside_the_run(tmp_path, monkeypatch):
    """MuJoCo's own handler appends every warning to MUJOCO_LOG.TXT in the
    directory the run is in, which is the plugin's package."""
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(simulate_mujoco.Diverged):
        simulate_mujoco.process(str(tmp_path), {"scene_file": _diverging(tmp_path), "duration": 1.0})

    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()


def test_without_the_check_the_diverged_run_never_ends_and_its_state_is_garbage(tmp_path):
    """Why the check is there. MuJoCo says it once, on its console, and goes on:
    the clock stops advancing -- so 'while time < duration' never ends -- and
    the state it is left in is a finite number that means nothing, which no
    check for NaN would catch. A thousand steps here, rather than the loop the
    plugin used to run, which is the hang this proves."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(_diverging(tmp_path))
    data = mujoco.MjData(model)
    console = simulate_mujoco.Console(mujoco)
    console.listen()  # rather than a MUJOCO_LOG.TXT beside the tests
    try:
        mujoco.mj_forward(model, data)
        for _ in range(1000):
            mujoco.mj_step(model, data)
    finally:
        console.close()

    assert data.warning.number[int(mujoco.mjtWarning.mjWARN_BADQACC)] > 0
    assert data.time < 1.0  # a thousand steps of 10 ms, and not one second gone
    assert math.isfinite(float(data.qpos[0]))


# --------------------------------------------------------------------------- #
# A run that will not end                                                     #
# --------------------------------------------------------------------------- #


def test_a_run_that_will_not_finish_within_its_budget_stops_itself_saying_how_far_it_got(tmp_path):
    pytest.importorskip("mujoco")
    scene = _write(tmp_path, FALLS)
    started = time.monotonic()

    # Ten thousand seconds of simulated time is five million steps -- tens of
    # seconds on any machine -- and the budget is one and a half.
    with pytest.raises(simulate_mujoco.OutOfTime) as raised:
        simulate_mujoco.process(str(tmp_path), {"scene_file": scene, "duration": 10000.0, "timeout": 1.5})

    assert time.monotonic() - started < 1.5  # before PartCAD would have stopped it
    message = str(raised.value)
    assert "of the 10000 s asked for" in message
    assert "budget is 1.5 s" in message


def test_the_budget_leaves_a_margin_for_saying_so():
    guard = simulate_mujoco.Guard({"timeout": 300}, 100, started=0.0)
    assert guard.deadline == 270.0
    guard = simulate_mujoco.Guard({"timeout": 3600}, 100, started=0.0)
    assert guard.deadline == 3570.0
    guard = simulate_mujoco.Guard({"timeout": 0.5}, 100, started=0.0)
    assert guard.deadline == 0.25


def test_with_no_budget_only_the_steps_bound_the_run():
    """An older PartCAD hands no 'timeout'; nothing bounds the wall clock then."""
    guard = simulate_mujoco.Guard({}, 100)
    guard.check(210, 1.0, 1.0)

    with pytest.raises(simulate_mujoco.OutOfTime, match="took 211 steps"):
        guard.check(211, 0.5, 1.0)


def test_a_run_taken_back_in_time_is_stopped_by_its_step_count(tmp_path, monkeypatch):
    """What a model that diverges again and again looks like without the
    divergence check: MuJoCo resets it to nought seconds every time, and the run
    never reaches its end."""
    mujoco = pytest.importorskip("mujoco")
    scene = _write(tmp_path, FALLS)
    monkeypatch.setattr(simulate_mujoco, "diverged", lambda *args: None)
    real = mujoco.mj_step

    def stuck(model, data):
        real(model, data)
        data.time = 0.0

    monkeypatch.setattr(mujoco, "mj_step", stuck)

    with pytest.raises(simulate_mujoco.OutOfTime, match="without reaching 0.1 s"):
        simulate_mujoco.process(str(tmp_path), {"scene_file": scene, "duration": 0.1})


# --------------------------------------------------------------------------- #
# A run asked for something absurd                                            #
# --------------------------------------------------------------------------- #

MODEL = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002))


@pytest.mark.parametrize(
    "request_, named",
    [
        ({"duration": 0}, "'duration'"),
        ({"duration": -1.0}, "'duration'"),
        ({"duration": float("nan")}, "'duration'"),
        ({"duration": "ten"}, "'duration'"),
        ({"duration": True}, "'duration'"),
        ({"timestep": 0}, "'timestep'"),
        ({"timestep": 5.0}, "'timestep'"),
        ({"duration": 1e9}, "steps"),
        ({"duration": 1.0, "timestep": 1e-9}, "steps"),
        ({"samples": -1}, "'samples'"),
        ({"samples": 2.5}, "'samples'"),
        ({"samples": 1e6}, "'samples'"),
        ({"duration": 0.01, "samples": 100}, "'samples'"),
    ],
)
def test_an_absurd_run_is_refused_before_anything_is_stepped(request_, named):
    with pytest.raises(ValueError) as raised:
        simulate_mujoco.run_parameters(request_, MODEL)
    assert named in str(raised.value)


def test_a_sensible_run_is_what_it_asked_for():
    assert simulate_mujoco.run_parameters({}, MODEL) == (10.0, 0.002, 5000, 0)
    assert simulate_mujoco.run_parameters({"duration": 2, "timestep": 0.001, "samples": 4.0}, MODEL) == (
        2.0,
        0.001,
        2000,
        4,
    )


def test_a_refused_run_never_reaches_mujoco(tmp_path):
    pytest.importorskip("mujoco")
    scene = _write(tmp_path, FALLS)

    with pytest.raises(ValueError, match="'duration'"):
        simulate_mujoco.process(str(tmp_path), {"scene_file": scene, "duration": -5})


# --------------------------------------------------------------------------- #
# What MuJoCo warned about without stopping                                   #
# --------------------------------------------------------------------------- #


def test_a_warning_that_does_not_end_the_run_is_reported_beside_it():
    mujoco = pytest.importorskip("mujoco")
    numbers = [0] * 8
    numbers[int(mujoco.mjtWarning.mjWARN_CONTACTFULL)] = 3
    data = types.SimpleNamespace(warning=types.SimpleNamespace(number=numbers))

    (said,) = simulate_mujoco.degraded(mujoco, data)

    assert "contacts" in said and "mjWARN_CONTACTFULL" in said and "3 times" in said


def test_a_well_behaved_run_has_nothing_to_say(tmp_path):
    pytest.importorskip("mujoco")
    scene = _write(tmp_path, FALLS)

    result = simulate_mujoco.process(str(tmp_path), {"scene_file": scene, "duration": 0.5, "timeout": 60})

    assert result["success"] is True
    assert "warnings" not in result
    assert result["after"]["bodies"]["block"]["pos"][2] < 20.0  # it fell, in millimetres
