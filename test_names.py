#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""PartCAD's names in a model MuJoCo can load, and back in what a run reports.

MJCF holds nearly every name PartCAD gives -- MuJoCo's compiler accepts any
characters, and asks only that two elements of one type differ -- except a
body called 'world', which is the '<worldbody>' itself. The exporter writes
such a body under another name by MJCF's own rules ('export_mjcf.MJCF_NAMES'
and 'MJCF_BODY_NAMES', through PartCAD's 'wrappers/naming.py'), says which in a
sidecar beside the model, and the run reports under PartCAD's names.

Needs PartCAD, and MuJoCo for the runs, as 'test_export_joints.py' does.
"""

import os
import sys

import pytest

import partcad as pc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(pc.__file__), "wrappers"))

import kinematics  # noqa: E402
import naming  # noqa: E402
from test_export_joints import export, export_mjcf, in_a_scene, rig, stand_in_geometry, weighed  # noqa: E402,F401

ARM_CALLED_WORLD = """
links:
  - part: base
  - part: arm
    name: world
    connect: {name: base, with: pin, to: bore}
"""


def scene_of(*labels):
    """A scene called 'world' of free parts with these labels, side by side."""
    return {
        "name": "//p:world",
        "label": "world",
        "kind": kinematics.SCENE,
        "assembly": [
            {
                "name": "//p:%d" % index,
                "label": name,
                "kind": kinematics.PART,
                "brep": b"%d" % index,
                "location": [[30.0 * index, 0.0, 100.0], [0.0, 0.0, 1.0], 0.0],
            }
            for index, name in enumerate(labels)
        ],
    }


def run(path, duration=0.05):
    pytest.importorskip("mujoco")
    import simulate_mujoco

    return simulate_mujoco.process(os.path.dirname(str(path)), {"scene_file": str(path), "duration": duration})


def test_mjcf_reserves_world_for_a_body_and_nothing_else():
    import export_mjcf as module

    assert not module.MJCF_BODY_NAMES.valid("world")
    assert module.MJCF_NAMES.valid("world")
    # What MuJoCo's compiler accepts as it is, which the transcoder leaves alone.
    for name in ("//p/x:y z", "колесо", "Cube", "cube-2", "nut-bolt.turnZ"):
        assert module.MJCF_BODY_NAMES.valid(name), name
    assert not module.MJCF_NAMES.valid("a\nb")


def test_a_body_called_world_is_written_under_another_name_and_the_sidecar_says_so(tmp_path, export_mjcf):
    result, path, mujoco = export(export_mjcf, tmp_path, scene_of("world", "cube"), static=False, names_file=True)

    # The scene keeps its name: MJCF reserves 'world' for a body, not a model.
    assert mujoco.get("model") == "world"
    assert [body.get("name") for body in mujoco.findall("worldbody/body")] == ["world-2", "cube"]
    assert any("'world' as 'world-2'" in warning for warning in result["warnings"])
    sidecar = naming.read_sidecar(str(path))
    assert sidecar["format"] == "mjcf"
    assert sidecar["names"]["body"] == {"world": "world-2", "cube": "cube"}
    assert sidecar["bodies"] == {"world": {"path": ["world-2"]}, "cube": {"path": ["cube"]}}


def test_no_sidecar_is_written_unless_asked_for(tmp_path, export_mjcf):
    _result, path, _mujoco = export(export_mjcf, tmp_path, scene_of("cube"))
    assert not os.path.exists(naming.sidecar_path(str(path)))


def test_a_run_reports_every_body_under_partcads_name(tmp_path, export_mjcf):
    _result, path, _mujoco = export(export_mjcf, tmp_path, scene_of("world", "cube"), static=False, names_file=True)
    result = run(path)
    for reading in (result["before"], result["after"]):
        assert sorted(reading["bodies"]) == ["cube", "world"]
    assert result["before"]["bodies"]["world"]["pos"] == pytest.approx([0.0, 0.0, 100.0])


def test_a_model_with_no_sidecar_is_reported_under_the_names_it_has(tmp_path, export_mjcf):
    _result, path, _mujoco = export(export_mjcf, tmp_path, scene_of("world", "cube"), static=False)
    assert sorted(run(path)["after"]["bodies"]) == ["cube", "world-2"]


def test_a_mechanism_reports_its_bodies_and_joints_under_partcads_names(tmp_path, stand_in_geometry, export_mjcf):
    tree = in_a_scene(rig(tmp_path, "motion: revolute", ARM_CALLED_WORLD), [[0.0, 0.0, 500.0], [1.0, 0.0, 0.0], 90.0])
    _result, path, mujoco = export(export_mjcf, tmp_path, tree, static=False, ground_plane=False, names_file=True)
    assert [body.get("name") for body in mujoco.iter("body")] == ["subject", "world-2"]
    assert [joint.get("name") for joint in mujoco.iter("joint")] == ["world-base"]
    sidecar = naming.read_sidecar(str(path))
    assert sidecar["joints"] == {"world-base": {"path": ["world-base"], "joint": "world-base"}}

    result = run(path, 0.2)
    for reading in (result["before"], result["after"]):
        assert sorted(reading["bodies"]) == ["subject", "world"]
        assert sorted(reading["joints"]) == ["world-base"]
    # In the world's frame, where PartCAD's own joint puts the arm at the angle
    # MuJoCo reports (the body read a step behind the joint; see 'joints').
    found = kinematics.bodies(tree)
    arm = found.by_name("world")
    angle = result["before"]["joints"]["world-base"]["pos"]
    expected = kinematics.compose(arm.parent.world, arm.joint.location([angle]))
    assert result["before"]["bodies"]["world"]["pos"] == pytest.approx(list(expected[1]), abs=1e-6)


def test_what_the_sidecar_does_not_mention_is_reported_as_it_is():
    import simulate_mujoco

    names = simulate_mujoco.Names({"bodies": {"world": {"path": ["world-2"]}}, "joints": {"j": {"path": ["j-2"]}}})
    assert (names.body("world-2"), names.joint("j-2"), names.body("other")) == ("world", "j", "other")
    assert simulate_mujoco.Names.beside("/nowhere/scene.xml").bodies == {}
