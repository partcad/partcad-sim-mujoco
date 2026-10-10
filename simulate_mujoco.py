#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The MuJoCo simulation plugin (see 'partcad.yaml' beside this file).

Runs a scene under gravity for a while and says where everything ended up.

The scene arrives as MJCF -- PartCAD exported it, with every body free to move
and a ground plane under it, under the scene's gravity and in the fluid the
scene is filled with, if it says either -- so all this does is load the model,
step it, and
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
"""

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

# The '<custom><numeric>' a buoyed body's centre of buoyancy is written under
# by this package's exporter, followed by the body's name: three numbers, in
# metres, in the body's own frame. 'export_mjcf.CENTRE_OF_BUOYANCY_PREFIX' is
# its twin; a test keeps the two the same.
CENTRE_OF_BUOYANCY_PREFIX = "partcad:centre_of_buoyancy:"

# MuJoCo's own viewer shows geom groups 0 to 2 and hides the rest, and models
# put what is only there to collide - a convex hull, a simplified proxy - in a
# group above those. The snapshots show what MuJoCo's viewer would.
VISIBLE_GROUPS = 3


def centres_of_buoyancy(mujoco, model):
    """(body id, centre of buoyancy in the body's frame) for every body the exporter buoyed.

    The lift itself is the body's ``gravcomp`` -- MuJoCo applies it at the
    centre of mass, which is where it belongs for a body of one material. A
    body whose centre of mass is not its centre of volume (a keel, a ballast)
    also needs the moment that lift has about its centre of mass, and MuJoCo has
    no way to apply a force anywhere else; 'apply_buoyancy_moments' does it from
    what this returns. Empty for a model with no fluid, or one PartCAD did not
    write.
    """
    import numpy

    found = []
    for index in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, index) or ""
        if not name.startswith(CENTRE_OF_BUOYANCY_PREFIX):
            continue
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name[len(CENTRE_OF_BUOYANCY_PREFIX) :])
        start, size = model.numeric_adr[index], model.numeric_size[index]
        if body < 0 or size != 3:
            continue
        found.append((body, numpy.array(model.numeric_data[start : start + size], dtype=float)))
    return found


def apply_buoyancy_moments(model, data, centres):
    """Turn each buoyed body by the moment its lift has about its centre of mass.

    The lift is what ``gravcomp`` applies at the centre of mass:
    -gravcomp * mass * gravity. Acting at the centre of buoyancy instead, it
    adds the moment (r_buoyancy - r_mass) x lift, which is what rights a body
    whose ballast is below its middle and capsizes one whose ballast is above.
    Set every step, before MuJoCo steps, from where the body is now; it is a
    torque alone, so the lift is not applied twice.
    """
    import numpy

    gravity = numpy.array(model.opt.gravity, dtype=float)
    for body, local in centres:
        lift = -model.body_gravcomp[body] * model.body_mass[body] * gravity
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
        data.xfrc_applied[body, 3:6] = numpy.cross(point - data.xipos[body], lift)


def snapshot(mujoco, model, data):
    """Where every body is right now, keyed by the name the MJCF gave it.

    The world body is left out: it is body 0 of every model, it is the frame
    everything else is stated in, and it never moves. What is left is exactly
    the bodies PartCAD's exporter wrote out of the scene, under the names it
    gave them - which is what makes a validation expression readable.
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
    return {"time": float(data.time), "bodies": bodies}


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
        # The exported model already carries the scene's gravity, or MuJoCo's
        # own when the scene states none. This is only here when the 'simulate:'
        # that asked for the run passed one in 'params' -- this package's
        # declaration deliberately states no default, because a default here
        # would override every scene's -- and it is the explicit, per-run answer
        # that beats the scene's. A body's buoyancy is a fraction of its weight
        # ('gravcomp'), so it follows the new gravity without anything else
        # being rewritten.
        model.opt.gravity[:] = [float(v) for v in gravity]

    data = mujoco.MjData(model)
    # Positions the bodies where the model says they are and computes
    # everything derived from that, without advancing time: this is the state
    # the scene described, which is what 'before' has to be.
    mujoco.mj_forward(model, data)
    before = snapshot(mujoco, model, data)
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
    centres = centres_of_buoyancy(mujoco, model)
    while data.time < duration:
        if centres:
            apply_buoyancy_moments(model, data, centres)
        mujoco.mj_step(model, data)
        steps += 1
        if next_sample is not None and data.time >= next_sample:
            trace.append(snapshot(mujoco, model, data))
            next_sample += duration / (samples + 1)

    after = snapshot(mujoco, model, data)
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
    if model.opt.density > 0 or model.opt.viscosity > 0:
        # The fluid the scene was filled with, as MuJoCo ran it: kg/m^3 and Pa*s,
        # MuJoCo's own units, the way 'gravity' beside it is in m/s^2. Left out
        # for a run in a vacuum, which is what a result has always meant.
        result["medium"] = {"density": float(model.opt.density), "viscosity": float(model.opt.viscosity)}
    if trace:
        result["samples"] = trace
    drawn = take_snapshots(path, pictures, looks, warnings)
    if drawn:
        result["snapshots"] = drawn
    if warnings:
        result["warnings"] = warnings
    return result
