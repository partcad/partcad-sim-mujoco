#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The MuJoCo simulation plugin (see 'partcad.yaml' beside this file).

Runs a scene under gravity for a while and says where everything ended up.

The scene arrives as MJCF -- PartCAD exported it, with the bodies its body rule
finds, every one nothing holds free to move and every joint between them
written, a ground plane under it, under the scene's gravity and in the fluid the
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

**Names are PartCAD's.** A validation reads ``after["bodies"]["cube"]`` and
``after["joints"]["swing"]`` by the names PartCAD gave them, and the model may
not hold every one of those as it is: MJCF has a body called 'world' of its own,
so a scene's body of that name is written under another. The exporter says which
in a sidecar beside the model (PartCAD's 'wrappers/naming.py'), and every body,
joint and message here is reported under the PartCAD name it reads there; see
'Names'. A model with no sidecar is reported under the names it has.

A model with joints in it says one more thing, and the reading says it too:
where each joint is and how fast it is moving, as ``joints`` beside ``bodies``,
in the terms of the ``motion:`` an interface declares -- degrees for a turn,
millimetres for a move. A joint is reported once, under PartCAD's name for it,
however many MJCF joints it had to be written as and from whatever zero: the
sidecar carries each one's joint map, and PartCAD's own 'kinematics.reading()'
turns MuJoCo's values back into PartCAD's. See 'joints' below.

**A run that cannot be trusted is not reported as one.** MuJoCo does not stop a
simulation whose state stops being a number: it says so once, on its console,
and *resets the state* to where the model started -- so the run goes on, the
reading at the end is of a model that never moved, and a validation that says
"nothing fell over" passes. And a run that keeps diverging keeps being reset to
nought seconds, and never reaches the end. So the run is watched: a state that
diverges stops it with the time and the joint and body it diverged at (see
'diverged'), a run is refused before it starts when what it is asked for is
absurd (see 'run_parameters'), and it is bounded twice over -- by the number of
steps its duration takes, and by the wall-clock budget PartCAD hands it as
``timeout``, which it stops a little short of, to say how far it got before
PartCAD stops it for it (see 'Guard').
"""

import math
import os
import sys
import time

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

# The '<custom><numeric>' a buoyed body's centre of buoyancy is written under
# by this package's exporter, followed by the body's name: three numbers, in
# metres, in the body's own frame. 'export_mjcf.CENTRE_OF_BUOYANCY_PREFIX' is
# its twin; a test keeps the two the same.
CENTRE_OF_BUOYANCY_PREFIX = "partcad:centre_of_buoyancy:"

# The '<keyframe><key>' this package's exporter writes where a model cannot be
# written at its start -- a bounded ball that starts away from its zero -- and
# which a run therefore starts from. 'export_mjcf.START_KEY' is its twin; a
# test keeps the two the same.
START_KEY = "partcad:start"

# What a run is when nothing says otherwise, in seconds of simulated time.
DEFAULT_DURATION = 10.0

# The most steps one run may take, and the most readings it may report. Each is
# far beyond anything a validation asks for -- fifty million steps is more than a
# day of simulated time at MuJoCo's own two-millisecond step -- and each is what
# a typo in 'duration', 'timestep' or 'samples' turns into: a run that steps for
# hours, or a result of a few gigabytes. Refused before anything is stepped.
MAX_STEPS = 50_000_000
MAX_SAMPLES = 10_000

# The longest step taken: a step of more than a second is a typo for one of a
# millisecond, and MuJoCo would integrate it -- for one step, before diverging.
MAX_TIMESTEP = 1.0

# How much of the wall-clock budget PartCAD hands the run ('timeout') is left
# unused, so that the run stops itself and says how far it got before PartCAD
# stops it and can say only that it did not finish: a tenth of it, at least one
# second and at most thirty.
BUDGET_MARGIN = 0.1
BUDGET_MARGIN_MIN = 1.0
BUDGET_MARGIN_MAX = 30.0

# The warnings MuJoCo raises when the state it integrates stops being a number
# -- not a number at all, infinite, or past 1e10 -- and what each one is of. It
# resets the state when it raises one, and goes on; see 'diverged'. Named
# rather than numbered, because the numbering is MuJoCo's.
DIVERGING = (
    ("mjWARN_BADQPOS", "position", "qpos"),
    ("mjWARN_BADQVEL", "velocity", "dof"),
    ("mjWARN_BADQACC", "acceleration", "dof"),
)

# The warnings that do not end the run, and what each says about its answer.
# Reported in 'warnings', because a run that hit one is a run whose answer is
# not quite what it looks like.
DEGRADING = {
    "mjWARN_CONTACTFULL": "MuJoCo ran out of room for contacts, so some contacts were not made",
    "mjWARN_CNSTRFULL": "MuJoCo ran out of room for constraints, so some constraints were not applied",
    "mjWARN_INERTIA": "a body's inertia is not physical, so MuJoCo could not move it as the model says",
    "mjWARN_BADCTRL": "a control was not a finite number, so MuJoCo set it to zero",
}

# The remedies for a model that diverges, in the order they are worth trying.
DIVERGED_REMEDY = (
    "A smaller 'timestep', an integrator that is more stable for stiff models ('implicitfast'), and a mass or an"
    " inertia that is not near zero are the usual remedies"
)


class Diverged(Exception):
    """MuJoCo's state stopped being a number, so nothing after that instant is a reading."""


class OutOfTime(Exception):
    """The run used up the time it was allowed, in steps or in wall-clock seconds."""


def run_parameters(request, model):
    """The run's duration, timestep and number of samples, or a ValueError saying which is absurd.

    'timestep' is the request's, else the model's own. Refused rather than
    clamped: a run asked for something it cannot do should say so, and one that
    quietly did something else would be validated against a run nobody asked
    for.
    """

    def number(name, value):
        if isinstance(value, bool):
            raise ValueError("'%s' is %r, which is not a number" % (name, value))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("'%s' is %r, which is not a number" % (name, value))
        if not math.isfinite(value):
            raise ValueError("'%s' is %r, which is not a finite number" % (name, value))
        return value

    duration = request.get("duration")
    duration = DEFAULT_DURATION if duration is None else number("duration", duration)
    if duration <= 0:
        raise ValueError(
            "'duration' is %g s: a run has to last for some time, or 'before' and 'after' are one reading" % duration
        )

    timestep = request.get("timestep")
    timestep = float(model.opt.timestep) if timestep is None else number("timestep", timestep)
    if timestep <= 0 or timestep > MAX_TIMESTEP:
        raise ValueError(
            "'timestep' is %g s: it has to be more than nought and at most %g s (MuJoCo's own is 0.002 s)"
            % (timestep, MAX_TIMESTEP)
        )

    steps = math.ceil(duration / timestep)
    if steps > MAX_STEPS:
        raise ValueError(
            "'duration' of %g s at a 'timestep' of %g s is %d steps, more than the %d a run may take."
            " Shorten the duration or lengthen the step" % (duration, timestep, steps, MAX_STEPS)
        )

    samples = request.get("samples")
    samples = 0 if samples is None else number("samples", samples)
    if samples != int(samples) or samples < 0:
        raise ValueError("'samples' is %g: it is a number of readings, a whole number from nought up" % samples)
    samples = int(samples)
    if samples > min(MAX_SAMPLES, steps):
        raise ValueError(
            "'samples' is %d, more than the %d readings a run of %d steps can report"
            % (samples, min(MAX_SAMPLES, steps), steps)
        )
    return duration, timestep, steps, samples


def budget_of(request):
    """The wall-clock seconds PartCAD gives the run ('timeout'), or None for no budget."""
    value = request.get("timeout")
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


class Guard:
    """What stops a run that will not end: a count of steps and a wall-clock deadline.

    The steps are what the duration takes, and twice that is the bound: a run
    that has taken more has been taken back to an earlier time, which is what
    MuJoCo does to a model that diverges. 'diverged' is the precise net; this
    one is under it.

    The deadline is the budget PartCAD gave the run, less 'BUDGET_MARGIN', from
    when this script started. PartCAD stops the run itself at the budget -- this
    is the second line, and the one that can say how far the run got.
    """

    def __init__(self, request, steps, started=None):
        started = time.monotonic() if started is None else started
        budget = budget_of(request)
        if budget is None:
            self.budget = self.deadline = None
        else:
            margin = min(BUDGET_MARGIN_MAX, max(BUDGET_MARGIN_MIN, BUDGET_MARGIN * budget))
            self.budget = budget
            self.deadline = started + (budget - margin if budget > margin else budget / 2.0)
        self.started = started
        self.planned = steps
        self.limit = 2 * steps + 10

    def check(self, steps, simulated, duration):
        """Raise 'OutOfTime' when the run has taken too many steps or too long."""
        if steps > self.limit:
            raise OutOfTime(
                "MuJoCo took %d steps without reaching %g s (the duration takes %d); it reached %g s"
                % (steps, duration, self.planned, simulated)
            )
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise OutOfTime(
                "MuJoCo reached %g s of the %g s asked for in %.1f s of wall-clock time, and the run's budget is %g s:"
                " it was stopped so that it could say so. Give it longer with 'timeout' in the simulation's"
                " 'params:', or shorten 'duration'"
                % (simulated, duration, time.monotonic() - self.started, self.budget)
            )


class Names:
    """PartCAD's name for what the model calls each body and each joint, and what each joint is.

    Read out of the sidecar the exporter wrote beside the model (see
    'naming.read_sidecar' in PartCAD): its 'bodies' and 'joints' say where each
    PartCAD body and reported joint is, and the last name of that path is what
    MuJoCo calls it. A name the sidecar does not mention -- every name of a model
    with no sidecar -- is reported as it is.

    A joint entry also carries its joint map ('"map"', which PartCAD's
    'kinematics.joint_map()' wrote): which PartCAD joint the MJCF joint is
    written for, which of its parameters, in what units and from what start.
    'mapped' is those, by MJCF name, and 'joints()' below reads them back.
    """

    def __init__(self, sidecar=None):
        sidecar = sidecar or {}
        self.bodies = self._reverse(sidecar.get("bodies"))
        self.joints = self._reverse(sidecar.get("joints"))
        # PartCAD body -> what MuJoCo calls it: where a floating joint's
        # reading finds the body it is read against.
        self.internal_bodies = {partcad: internal for internal, partcad in self.bodies.items()}
        # MJCF joint -> (the PartCAD joint it is written for, its map).
        self.mapped = {}
        for name, entry in (sidecar.get("joints") or {}).items():
            path = entry.get("path") if isinstance(entry, dict) else None
            mapping = entry.get("map") if isinstance(entry, dict) else None
            if isinstance(path, list) and path and isinstance(mapping, dict):
                self.mapped[str(path[-1])] = (str(entry.get("joint") or mapping.get("joint") or name), mapping)

    @staticmethod
    def _reverse(entries):
        out = {}
        for name, entry in (entries or {}).items():
            path = entry.get("path") if isinstance(entry, dict) else None
            if isinstance(path, list) and path:
                out[str(path[-1])] = name
        return out

    @classmethod
    def beside(cls, scene_file):
        try:
            import naming
        except ImportError:
            # A PartCAD from before names were transcoded, which wrote none.
            return cls()
        return cls(naming.read_sidecar(scene_file))

    def body(self, internal):
        return self.bodies.get(internal, internal)

    def joint(self, internal):
        return self.joints.get(internal, internal)

    def map(self, internal):
        """(PartCAD joint, joint map) of the MJCF joint 'internal', or None where the sidecar has no map of it."""
        return self.mapped.get(internal)


# Every name as MuJoCo has it: a model with no sidecar.
AS_WRITTEN = Names()


def warning_index(mujoco, name):
    """MuJoCo's number for one of its warnings, or None for a MuJoCo that has no such warning."""
    value = getattr(mujoco.mjtWarning, name, None)
    return int(value) if value is not None else None


def diverged(mujoco, model, data, when, names=AS_WRITTEN):
    """Raise 'Diverged' if MuJoCo has said that the state stopped being a number.

    'when' is the simulated time the step started at, which is the last instant
    the state was one: MuJoCo has reset it by the time this is asked. Named by
    the joint it diverged at and the body that joint moves, which is what a
    reader can find in the model.
    """
    numbers = data.warning.number
    for name, quantity, kind in DIVERGING:
        index = warning_index(mujoco, name)
        if index is None or not numbers[index]:
            continue
        where = int(data.warning.lastinfo[index])
        raise Diverged(
            "MuJoCo's simulation diverged at t = %g s: the %s of %s is not a finite number (%s). "
            "MuJoCo resets a model that does this and goes on, so nothing after this instant is a reading. %s"
            % (when, quantity, located(mujoco, model, kind, where, names), name, DIVERGED_REMEDY)
        )


def located(mujoco, model, kind, index, names=AS_WRITTEN):
    """'joint <j> (body <b>)' for a degree of freedom or a position coordinate, by PartCAD's names for them."""
    joint = body = None
    if kind == "dof" and 0 <= index < model.nv:
        joint, body = int(model.dof_jntid[index]), int(model.dof_bodyid[index])
    elif kind == "qpos" and 0 <= index < model.nq:
        for candidate in range(model.njnt):
            if model.jnt_qposadr[candidate] <= index:
                joint = candidate
        if joint is not None:
            body = int(model.jnt_bodyid[joint])
    if joint is None:
        return "coordinate %d" % index
    joint_name = names.joint(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "joint_%d" % joint)
    body_name = names.body(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "body_%d" % body)
    return "joint '%s' (body '%s')" % (joint_name, body_name)


def degraded(mujoco, data):
    """What MuJoCo warned about that did not end the run, as sentences for 'warnings'."""
    said = []
    for name, meaning in DEGRADING.items():
        index = warning_index(mujoco, name)
        if index is not None and data.warning.number[index]:
            said.append("%s (%s, %d times)" % (meaning, name, int(data.warning.number[index])))
    return said


def not_finite(data):
    """Whether the state holds a number that is not one, which MuJoCo's own checks may have let by."""
    import numpy

    return not (numpy.isfinite(data.qpos).all() and numpy.isfinite(data.qvel).all())


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


def snapshot(mujoco, model, data, scratch, names=AS_WRITTEN):
    """Where every body and every joint is right now, by PartCAD's names for them.

    The world body is left out: it is body 0 of every model, it is the frame
    everything else is stated in, and it never moves. What is left is exactly
    the bodies PartCAD's exporter wrote out of the scene, under the names
    PartCAD gave them ('names') - which is what makes a validation expression
    readable. Every pose is in the world frame, MuJoCo's 'xpos' and 'xquat'.

    'scratch' is an 'MjData' of the same model that 'joints' may overwrite, or
    None for a model that has no joint to report.
    """
    bodies = {}
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if not name:
            name = "body_%d" % index
        bodies[names.body(name)] = {
            "pos": [float(v) * MM_PER_M for v in data.xpos[index]],
            "quat": [float(v) for v in data.xquat[index]],
        }
    return {"time": float(data.time), "bodies": bodies, "joints": joints(mujoco, model, data, scratch, names)}


def joints(mujoco, model, data, scratch, names=AS_WRITTEN):
    """Where every joint is and how fast it moves, keyed by PartCAD's name for it ('names').

    Stated in PartCAD's vocabulary rather than MuJoCo's: the one the 'motion:'
    of an interface is written in, which is what a joint in a model PartCAD
    exported came from, and what a validation of it is written against.

    **A model PartCAD exported** is read through the joint maps in its sidecar
    (see 'Names'), with PartCAD's own 'kinematics.reading()' -- so no copy of
    how a joint was written lives here. Every MJCF joint written for one
    PartCAD joint is read back as that one joint, under its name, however many
    it took:

      type      what 'kinematics' classifies the joint as: 'revolute',
                'prismatic', 'cylindrical', 'screw', 'universal', 'ball',
                'planar', 'floating' or 'composite'
      pos       its value as PartCAD states it, in degrees for a turn and
                millimetres for a move: a number for a joint of one freedom,
                and an object keyed by parameter for one of several
                ('{"turnZ": 30.0, "moveZ": 0.83}'); for a ball and a free body
                the turns their rotation decomposes into, where it can
      quat      a ball's and a floating joint's rotation from PartCAD's zero,
                'w x y z', in the joint's own frame
      vel       the same as 'pos', per second -- for a ball its angular
                velocity '[x, y, z]' about its moving frame, and for a
                floating joint its six rates by parameter beside 'angular',
                its angular velocity against the body it was connected to,
                '[x, y, z]' about its moving frame as a ball's is
      effort    what the model's actuators exert along each freedom, keyed as
                'vel' is: N*m for a turn, N for a move, and for a ball
                '[x, y, z]' N*m in the frame its 'vel' is in

    A hinge's and a slide's value is MuJoCo's own coordinate, since the
    exporter makes its 'ref' the starting value; a ball's is MuJoCo's
    rotation turned back by the start it was re-zeroed at; a floating joint's
    is its body's pose against the body it was connected to, and its rate the
    two bodies' velocities (see '_twist'). Its map says which.

    **A model with no sidecar** -- one somebody wrote by hand -- has no PartCAD
    joints to report, so each of its own joints is reported under its MJCF
    name, as MuJoCo has it:

      revolute    a hinge, limited or not: 'pos' in degrees, 'vel' in degrees
                  per second, 'effort' in N*m
      prismatic   a slide: 'pos' in millimetres, 'vel' in millimetres per
                  second, 'effort' in N
      ball        'quat' (w x y z), the turn away from where the model placed
                  the body; 'vel', the angular velocity [x, y, z] in degrees
                  per second about the axes of the body's own frame (MuJoCo's
                  own frame for it); 'effort', [x, y, z] in N*m in the same
                  frame

    'pos' is the joint's own coordinate, zero where the model placed the body
    -- or at the joint's 'ref', for a model that states one. A free joint is
    left out: its coordinate is exactly the body's position and orientation,
    which 'bodies' already states, in the same units.

    'effort' is what the model's actuators exert along the joint: MuJoCo's
    'qfrc_actuator', gear ratio and force limits applied, which is the quantity
    an interface's 'maxEffort' bounds. It is zero for a joint nothing drives,
    and that is every joint until a model has actuators. It is deliberately not
    the constraint force -- what a limit pushes back with, or what the joint
    transmits from one body to the other: those are reactions, a validation of
    whether a motor was strong enough asks about the action, and the constraint
    force on a degree of freedom mixes contacts, limits and equalities into one
    number that is none of them.

    A joint the MJCF did not name is reported as 'joint_<n>', 'n' being its
    index in the compiled model: the same rule 'bodies' follows, and the one
    name that does not depend on the order anything else was read in.
    """
    reported = reported_joints(mujoco, model, names)
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
    # what the bodies are read from -- and a floating joint, which is read
    # from two bodies' poses, is read from that state too, at 'time'.
    #
    # The bodies are not read from here, deliberately. They are read from the
    # run itself and so lag 'time' by up to a step, as they always have, and
    # moving them onto it changes what every existing validation is handed: a
    # change of its own, which would put a reading's bodies and its joints on
    # the same instant.
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)

    # Each name reported, in the order its first MJCF joint comes in the
    # model, with the MJCF joints it is read from: the several a PartCAD joint
    # was written as, or one of the model's own.
    groups = {}
    for index in reported:
        internal = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "joint_%d" % index
        mapped = names.map(internal)
        if mapped is None:
            groups[(names.joint(internal), None)] = [(index, None)]
        else:
            groups.setdefault((mapped[0], True), []).append((index, mapped[1]))

    reading = {}
    for (name, partcad), members in groups.items():
        if partcad:
            value = _partcad_reading(mujoco, model, scratch, members, names)
        else:
            value = _model_reading(mujoco, model, scratch, members[0][0])
        if value is not None:
            reading[name] = value
    return reading


def _model_reading(mujoco, model, state, index):
    """One joint of a model PartCAD did not write, as MuJoCo has it (see 'joints')."""
    kind = model.jnt_type[index]
    at, dof = model.jnt_qposadr[index], model.jnt_dofadr[index]
    qpos, qvel, effort = state.qpos, state.qvel, state.qfrc_actuator
    if kind == mujoco.mjtJoint.mjJNT_HINGE:
        return {
            "type": "revolute",
            "pos": float(qpos[at]) * DEG_PER_RAD,
            "vel": float(qvel[dof]) * DEG_PER_RAD,
            "effort": float(effort[dof]),
        }
    if kind == mujoco.mjtJoint.mjJNT_SLIDE:
        return {
            "type": "prismatic",
            "pos": float(qpos[at]) * MM_PER_M,
            "vel": float(qvel[dof]) * MM_PER_M,
            "effort": float(effort[dof]),
        }
    if kind == mujoco.mjtJoint.mjJNT_BALL:
        return {
            "type": "ball",
            "quat": [float(v) for v in qpos[at : at + 4]],
            "vel": [float(v) * DEG_PER_RAD for v in qvel[dof : dof + 3]],
            "effort": [float(v) for v in effort[dof : dof + 3]],
        }
    return None


def _pose(state, body):
    """A body's pose in the world, as 'kinematics.reading()' takes one: ((w, x, y, z), (x, y, z)), metres."""
    return (tuple(float(v) for v in state.xquat[body]), tuple(float(v) for v in state.xpos[body]))


def _twist(mujoco, model, state, body):
    """A body's velocity in the world, as 'kinematics.reading()' takes one: ((vx, vy, vz), (wx, wy, wz)).

    The linear velocity of the body *frame*'s origin in m/s, and the angular
    velocity in rad/s, both about the world's axes. That is MuJoCo's
    'mj_objectVelocity' of the body as 'mjOBJ_XBODY' -- its frame -- in the
    world's orientation, reordered: MuJoCo states the angular velocity first.
    Not 'mjOBJ_BODY', which is the body's centre of mass, moving faster than
    its frame's origin by the spin about it wherever the two are apart -- and
    'reading()' reads a free joint's pose, and so its rate, off the frame. The
    world body never moves.
    """
    if body == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    import numpy

    velocity = numpy.zeros(6)
    mujoco.mj_objectVelocity(model, state, mujoco.mjtObj.mjOBJ_XBODY, int(body), velocity, 0)
    return (tuple(float(v) for v in velocity[3:]), tuple(float(v) for v in velocity[:3]))


def _partcad_reading(mujoco, model, state, members, names):
    """One PartCAD joint, read from the MJCF joints it was written as and their maps (see 'joints').

    'members' are (MJCF joint index, joint map) pairs. What each map is is
    'kinematics.joint_map()''s and what to make of the values 'reading()''s;
    only MuJoCo's side is here: which of 'qpos' and 'qvel' each MJCF joint is,
    and for a floating joint the two bodies whose poses and velocities it is
    read from. None where a floating joint's other body is not in the model.
    """
    import kinematics

    maps, positions, velocities, efforts = [], [], [], {}
    ball = None
    for index, entry in members:
        kind = model.jnt_type[index]
        at, dof = model.jnt_qposadr[index], model.jnt_dofadr[index]
        maps.append(entry)
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            parent = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, names.internal_bodies.get(entry.get("parent"), entry.get("parent"))
            )
            if parent < 0:
                return None
            child = model.jnt_bodyid[index]
            positions.append((_pose(state, child), _pose(state, parent)))
            velocities.append((_twist(mujoco, model, state, child), _twist(mujoco, model, state, parent)))
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            positions.append([float(v) for v in state.qpos[at : at + 4]])
            velocities.append([float(v) for v in state.qvel[dof : dof + 3]])
            # In the frame the reading states the ball's angular velocity in:
            # its own, turned from the child body's that MuJoCo states it in.
            frame = (tuple(entry.get("frame") or (1.0, 0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
            ball = [float(v) for v in kinematics.rotate(frame, [float(v) for v in state.qfrc_actuator[dof : dof + 3]])]
        else:
            positions.append(float(state.qpos[at]))
            velocities.append(float(state.qvel[dof]))
            for coordinate in entry.get("coordinates") or []:
                efforts[coordinate["parameter"]] = float(state.qfrc_actuator[dof])

    result = kinematics.reading(maps, positions, velocities)
    if ball is not None:
        result["effort"] = ball
    elif efforts:
        one = maps[0].get("freedoms") == 1 and len(efforts) == 1
        result["effort"] = next(iter(efforts.values())) if one else efforts
    return result


def reported_joints(mujoco, model, names=AS_WRITTEN):
    """The joints 'joints' reports, by index: every one but a free one, and a free one a PartCAD joint is."""
    found = []
    for index in range(model.njnt):
        if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE:
            internal = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if not internal or names.map(internal) is None:
                continue
        found.append(index)
    return found


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


class Console:
    """What MuJoCo says while it runs, kept for the result rather than printed.

    MuJoCo's own handler prints a warning on standard output and appends it to
    'MUJOCO_LOG.TXT' in whatever directory the run is in -- which is the
    plugin's package, and for a package PartCAD fetched, its cached checkout.
    Kept here, the sentence MuJoCo wrote is what a divergence is reported with,
    and the file is never written.
    """

    LIMIT = 20

    def __init__(self, mujoco):
        self.mujoco = mujoco
        self.said = []
        self.previous = None
        self.listening = False

    def listen(self):
        get = getattr(self.mujoco, "get_mju_user_warning", None)
        put = getattr(self.mujoco, "set_mju_user_warning", None)
        if get is None or put is None:
            return
        self.previous = get()
        put(self.hear)
        self.listening = True

    def hear(self, message):
        message = str(message).strip()
        if message and message not in self.said and len(self.said) < self.LIMIT:
            self.said.append(message)

    def close(self):
        if self.listening:
            self.mujoco.set_mju_user_warning(self.previous)
            self.listening = False


def process(path, request):
    started = time.monotonic()
    import mujoco

    console = Console(mujoco)
    console.listen()
    try:
        return run(mujoco, path, request, started, console)
    except Diverged as e:
        if console.said:
            raise Diverged("%s. MuJoCo said: %s" % (str(e).rstrip("."), " ".join(console.said[:3])))
        raise
    finally:
        console.close()


def run(mujoco, path, request, started, console):
    """'process', with MuJoCo imported and what it says kept (see 'Console')."""
    scene_file = request["scene_file"]
    model = mujoco.MjModel.from_xml_path(scene_file)

    # Before anything is stepped: a run asked for something absurd is refused,
    # saying which parameter, rather than stepped for hours.
    duration, timestep, planned, samples = run_parameters(request, model)
    model.opt.timestep = timestep
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
    # Where the model starts. That is MuJoCo's own initial configuration --
    # every body where the file puts it -- unless the exporter wrote a start
    # the file could not put a body at: a ball bounded by a cone about its
    # zero that starts away from it, written at the zero and started here.
    # See 'write_ball' and 'write_keyframe' in 'export_mjcf.py'.
    start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, START_KEY)
    if start >= 0:
        mujoco.mj_resetDataKeyframe(model, data, start)
    # Positions the bodies where the model says they are and computes
    # everything derived from that, without advancing time: this is the state
    # the scene described, which is what 'before' has to be.
    mujoco.mj_forward(model, data)
    # What PartCAD calls everything the model holds, out of the sidecar beside
    # it, and what each of its joints is.
    names = Names.beside(scene_file)
    # Where the joints' efforts are worked out, so that working them out
    # touches neither the run nor what the bodies are read from; see 'joints'.
    # Not even made for a model with no joint to report -- a model of free
    # bodies, which is what a scene of parts is -- so that one costs exactly
    # what it did.
    scratch = mujoco.MjData(model) if reported_joints(mujoco, model, names) else None
    before = snapshot(mujoco, model, data, scratch, names)
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
    guard = Guard(request, planned, started)
    # A model that is unstable from the start is one MuJoCo already warned
    # about while working out where everything is.
    diverged(mujoco, model, data, data.time, names)
    while data.time < duration:
        if centres:
            apply_buoyancy_moments(model, data, centres)
        when = data.time
        mujoco.mj_step(model, data)
        steps += 1
        diverged(mujoco, model, data, when, names)
        guard.check(steps, data.time, duration)
        if next_sample is not None and data.time >= next_sample:
            trace.append(snapshot(mujoco, model, data, scratch, names))
            next_sample += duration / (samples + 1)

    if not_finite(data):
        # Past every check MuJoCo makes: nothing it warned about, and still not
        # a number. Not reported as a reading either.
        raise Diverged(
            "MuJoCo's simulation ended at t = %g s with a state that is not a finite number. %s"
            % (data.time, DIVERGED_REMEDY)
        )
    after = snapshot(mujoco, model, data, scratch, names)
    warnings.extend(degraded(mujoco, data))
    warnings.extend("MuJoCo said: %s" % message for message in console.said)
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
