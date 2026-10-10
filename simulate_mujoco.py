#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The MuJoCo simulation plugin (see 'partcad.yaml' beside this file).

Runs a scene under gravity for a while and says where everything ended up.

The scene arrives as MJCF -- PartCAD exported it, with every body free to move
and a ground plane under it -- so all this does is load the model, step it, and
take the same reading twice: once before anything has moved and once when the
time is up. That pair is what a ``simulate:``'s ``validation:`` expression is
handed, and it is the whole of what PartCAD requires a simulation plugin to
produce.

Positions are reported in **millimetres**, PartCAD's unit everywhere, not in the
metres MuJoCo works in. A validation expression is written by whoever wrote the
part, against the numbers that part is drawn in.

It also draws the two pictures PartCAD asks for -- the scene before and after,
from the viewpoint the request names -- out of the very model it steps; see
'snapshot_raster.py'. They never decide anything: a picture that cannot be
drawn is a warning beside the result.

Nothing here is specific to what is being simulated. A body is a body, and the
reading is "where is it and which way is it facing" -- which is all a static
description of a scene ever had to say, and so all that can be compared against
it. A plugin that needs to say more (a force, a temperature, a contact history)
states it beside 'before' and 'after' in its own vocabulary; see
'wrappers/wrapper_simulate.py' in PartCAD itself, which is what runs this.

A model with joints in it says one more thing, and the reading says it too:
where each joint is and how fast it is moving, as ``joints`` beside ``bodies``,
in the terms of the ``motion:`` an interface declares -- degrees for a turn,
millimetres for a move. See 'joints' below.
"""

import math
import os
import sys

# 'snapshot_raster' is this package's, beside this file. PartCAD runs this
# script by path, which puts nothing on sys.path for it.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Millimetres per metre: MJCF is metres by definition, PartCAD is millimetres
# throughout. Spelled out rather than imported from PartCAD's own
# 'urdf_common' because this script runs in a sandbox that carries MuJoCo and
# nothing else -- and because this package is not PartCAD and does not get to
# reach into it.
MM_PER_M = 1000.0

# Degrees per radian. MuJoCo computes every angle in radians, whatever the
# model's '<compiler angle=...>' said the file was written in; PartCAD states a
# turn in degrees, as the 'motion:' of an interface does, its limits included.
DEG_PER_RAD = 180.0 / math.pi

# MuJoCo's own viewer shows geom groups 0 to 2 and hides the rest, and models
# put what is only there to collide - a convex hull, a simplified proxy - in a
# group above those. The snapshots show what MuJoCo's viewer would.
VISIBLE_GROUPS = 3


def snapshot(mujoco, model, data, scratch):
    """Where every body and every joint is right now, by the names the MJCF gave them.

    The world body is left out: it is body 0 of every model, it is the frame
    everything else is stated in, and it never moves. What is left is exactly
    the bodies PartCAD's exporter wrote out of the scene, under the names it
    gave them - which is what makes a validation expression readable.

    'scratch' is an 'MjData' of the same model that 'joints' may overwrite, or
    None for a model that has no joint to report.
    """
    bodies = {}
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if not name:
            name = "body_%d" % index
        bodies[name] = {
            "pos": [float(v) * MM_PER_M for v in data.xpos[index]],
            "quat": [float(v) for v in data.xquat[index]],
        }
    return {"time": float(data.time), "bodies": bodies, "joints": joints(mujoco, model, data, scratch)}


def joints(mujoco, model, data, scratch):
    """Where every joint is and how fast it moves, keyed by the name the MJCF gave it.

    Stated in PartCAD's vocabulary rather than MuJoCo's: the one the 'motion:'
    of an interface is written in, which is what a joint in a model PartCAD
    exported came from, and what a validation of it is written against.

      revolute,   'pos' in degrees, 'vel' in degrees per second, 'effort' in
      continuous  N*m. Both are a MuJoCo hinge: 'revolute' when it is limited
                  and 'continuous' when it is not, which is the line PartCAD
                  (and URDF) draws between the two.
      prismatic   'pos' in millimetres, 'vel' in millimetres per second,
                  'effort' in N. A MuJoCo slide.
      ball        'quat' (w x y z), the turn away from where the model placed
                  the body; 'vel', the angular velocity [x, y, z] in degrees per
                  second about the axes of the body's own frame (MuJoCo's own
                  frame for it); 'effort', [x, y, z] in N*m in the same frame.

    'pos' is the joint's own coordinate, so it is zero where the model placed
    the body -- or the joint's 'ref', for a model that states one -- and not an
    angle against anything else. A pendulum written out horizontal reads 0
    there and a quarter turn, 90 one way or the other, hanging down.

    'effort' is what the model's actuators exert along the joint: MuJoCo's
    'qfrc_actuator', gear ratio and force limits applied, which is the quantity
    an interface's 'maxEffort' bounds. It is zero for a joint nothing drives,
    and that is every joint until a model has actuators. It is deliberately not
    the constraint force -- what a limit pushes back with, or what the joint
    transmits from one body to the other: those are reactions, a validation of
    whether a motor was strong enough asks about the action, and the constraint
    force on a degree of freedom mixes contacts, limits and equalities into one
    number that is none of them.

    A free joint is left out. It is how a body that is free to move is written
    -- which, in a scene PartCAD exported for a simulation, is every body -- and
    its coordinate is exactly the body's position and orientation, which
    'bodies' already states, in the same units.

    A joint the MJCF did not name is reported as 'joint_<n>', 'n' being its
    index in the compiled model: the same rule 'bodies' follows, and the one
    name that does not depend on the order anything else was read in.
    """
    reported = reported_joints(mujoco, model)
    if not reported:
        return {}

    # The state at 'time', with everything derived from it worked out afresh.
    #
    # After 'mj_step', MuJoCo holds the state the step ended at -- 'qpos',
    # 'qvel' and 'time' -- but everything it derives, the actuator forces
    # among them, from whatever it evaluated to take the step: the state the
    # step started from under the Euler integrators, and an intermediate
    # stage under RK4. Reading 'qfrc_actuator' as it stands would state an
    # effort from up to a step ago beside a position from now. So the state
    # is copied and evaluated on its own, which touches neither the run nor
    # what the bodies are read from.
    #
    # The bodies are not read from here, deliberately. They are read from the
    # run itself and so lag 'time' by up to a step, as they always have, and
    # moving them onto it changes what every existing validation is handed: a
    # change of its own, which would put a reading's bodies and its joints on
    # the same instant.
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)
    qpos, qvel, effort = scratch.qpos, scratch.qvel, scratch.qfrc_actuator

    reading = {}
    for index in reported:
        kind = model.jnt_type[index]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if not name:
            name = "joint_%d" % index
        at = model.jnt_qposadr[index]
        dof = model.jnt_dofadr[index]
        if kind == mujoco.mjtJoint.mjJNT_HINGE:
            reading[name] = {
                "type": "revolute" if model.jnt_limited[index] else "continuous",
                "pos": float(qpos[at]) * DEG_PER_RAD,
                "vel": float(qvel[dof]) * DEG_PER_RAD,
                "effort": float(effort[dof]),
            }
        elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
            reading[name] = {
                "type": "prismatic",
                "pos": float(qpos[at]) * MM_PER_M,
                "vel": float(qvel[dof]) * MM_PER_M,
                "effort": float(effort[dof]),
            }
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            reading[name] = {
                "type": "ball",
                "quat": [float(v) for v in qpos[at : at + 4]],
                "vel": [float(v) * DEG_PER_RAD for v in qvel[dof : dof + 3]],
                "effort": [float(v) for v in effort[dof : dof + 3]],
            }
    return reading


def reported_joints(mujoco, model):
    """The joints 'joints' reports, by index: every one but a free one."""
    return [index for index in range(model.njnt) if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE]


def geometry(mujoco, model, data):
    """What the scene looks like right now, in the terms 'snapshot_raster' draws.

    MuJoCo's own compiled geometry at MuJoCo's own poses: a mesh geom is the
    mesh MuJoCo loaded (re-centred on its own frame, which is the frame the geom
    is placed in), a primitive is tessellated, and a plane is the floor. Every
    geom of one body is one thing, so the outlines are drawn around bodies
    rather than around the pieces a body was written as.
    """
    import snapshot_raster

    solids, planes = [], []
    for index in range(model.ngeom):
        material = model.geom_matid[index]
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[index]
        if rgba[3] <= 0 or model.geom_group[index] >= VISIBLE_GROUPS:
            continue
        kind = model.geom_type[index]
        rotation = data.geom_xmat[index].reshape(3, 3)
        position = data.geom_xpos[index]
        size = model.geom_size[index]
        color = [float(v) for v in rgba[:3]]
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            planes.append({"origin": position.copy(), "axes": rotation.copy(), "color": color})
            continue
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[index]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            first, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            local = model.mesh_vert[start : start + count][model.mesh_face[first : first + faces]]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = snapshot_raster.box(size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            local = snapshot_raster.ellipsoid(size[0], size[0], size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            local = snapshot_raster.ellipsoid(size[0], size[1], size[2])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            local = snapshot_raster.cylinder(size[0], size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local = snapshot_raster.capsule(size[0], size[1])
        else:
            # A height field or an SDF: nothing a scene PartCAD wrote holds.
            continue
        solids.append(
            {
                "triangles": snapshot_raster.place(local, rotation, position),
                "color": color,
                "id": int(model.geom_bodyid[index]),
            }
        )
    return {"solids": solids, "planes": planes}


def look(mujoco, model, data, warnings):
    """'geometry', or None and a warning: reading it must not cost the run either."""
    try:
        return geometry(mujoco, model, data)
    except Exception as e:  # pylint: disable=broad-except
        warnings.append("the scene could not be read for the snapshots: %s: %s" % (type(e).__name__, e))
        return None


def take_snapshots(path, snapshot, scenes, warnings):
    """Draw the pictures PartCAD asked for, and say which were drawn.

    Never a reason for the run to fail: the verdict does not depend on a
    picture, so a picture that cannot be drawn is a warning beside a result
    rather than the loss of one.
    """
    # Both or neither: they are framed together, and one on its own is not a
    # comparison.
    if not snapshot or not path or any(scene is None for scene in scenes.values()):
        return {}
    try:
        import snapshot_raster

        return snapshot_raster.take(path, snapshot, scenes)
    except Exception as e:  # pylint: disable=broad-except
        warnings.append("the snapshots could not be drawn: %s: %s" % (type(e).__name__, e))
        return {}


def process(path, request):
    import mujoco

    scene_file = request["scene_file"]
    duration = float(request.get("duration") or 10.0)
    samples = int(request.get("samples") or 0)

    model = mujoco.MjModel.from_xml_path(scene_file)

    timestep = request.get("timestep")
    if timestep:
        model.opt.timestep = float(timestep)
    gravity = request.get("gravity")
    if gravity:
        # The exported MJCF already carries it; honouring it here too means a
        # 'simulate:' can ask for a different gravity without re-exporting.
        model.opt.gravity[:] = [float(v) for v in gravity]

    data = mujoco.MjData(model)
    # Positions the bodies where the model says they are and computes
    # everything derived from that, without advancing time: this is the state
    # the scene described, which is what 'before' has to be.
    mujoco.mj_forward(model, data)
    # Where the joints' efforts are worked out, so that working them out
    # touches neither the run nor what the bodies are read from; see 'joints'.
    # Not even made for a model with no joint to report -- a model of free
    # bodies, which is every scene PartCAD exports today -- so that one costs
    # exactly what it did.
    scratch = mujoco.MjData(model) if reported_joints(mujoco, model) else None
    before = snapshot(mujoco, model, data, scratch)
    # What PartCAD would like a picture of, and what the scene looks like now,
    # for the first of them. Read now because by the end it has moved.
    pictures = request.get("snapshot")
    warnings = []
    looks = {}
    if pictures:
        looks["before"] = look(mujoco, model, data, warnings)

    trace = []
    next_sample = duration / (samples + 1) if samples > 0 else None
    steps = 0
    while data.time < duration:
        mujoco.mj_step(model, data)
        steps += 1
        if next_sample is not None and data.time >= next_sample:
            trace.append(snapshot(mujoco, model, data, scratch))
            next_sample += duration / (samples + 1)

    after = snapshot(mujoco, model, data, scratch)
    if pictures:
        looks["after"] = look(mujoco, model, data, warnings)

    result = {
        "success": True,
        "before": before,
        "after": after,
        # Beside the two PartCAD requires: what this run actually was, so that a
        # report of a failed validation says what it was a validation of.
        "simulator": "mujoco",
        "version": getattr(mujoco, "__version__", None),
        "duration": duration,
        "timestep": float(model.opt.timestep),
        "steps": steps,
        "gravity": [float(v) for v in model.opt.gravity],
        "units": "mm",
    }
    if trace:
        result["samples"] = trace
    drawn = take_snapshots(path, pictures, looks, warnings)
    if drawn:
        result["snapshots"] = drawn
    if warnings:
        result["warnings"] = warnings
    return result
