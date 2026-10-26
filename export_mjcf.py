#
# PartCAD, 2026
#
# Licensed under Apache License, Version 2.0.
#
"""The MJCF (MuJoCo) exporter (declared under 'export:' in partcad.yaml beside this).

Writes a PartCAD scene -- or an assembly, or any other shape -- as an MJCF
``.xml`` model plus the mesh files it references. MJCF is what MuJoCo describes
a model in, and this is the format `pc sim` hands a scene over in: a simulation
plugin is given a file, and for the MuJoCo plugin that file is this one.

Like PartCAD's URDF exporter, this one writes the assembly *tree* rather than
the geometry it decodes to: PartCAD hands every exporter that tree as
``request["shape_tree"]``, beside the geometry in ``request["wrapped"]`` (see
'wrappers/wrapper_export.py' in PartCAD itself). The geometry keeps the tree's
shape and nothing else about it: every node's ``name`` and ``label`` is dropped
and its placement is baked in rather than staying readable as data, and the
bodies, their poses and their properties are built from all three.

A PartCAD released before 'shape_tree' hands the tree in 'wrapped' instead,
and only to an exporter that declares ``decode: false`` - which is why this one
still does, and why both are read: see 'tree_of()'.

Which nodes of the tree are bodies, and how the bodies hang together, is not
decided here. It is PartCAD's body rule, in PartCAD's own 'kinematics' (see
'wrappers/kinematics.py' in PartCAD, and section 7 of its simulation design
record): every element of a scene is a body of its own, everything inside an
assembly is attached -- welded into one body, or jointed to it by a connection
that keeps a degree of freedom -- and a scene element marked ``fixed: true`` is
welded to the world. This exporter writes what that rule returns:

  * the exported object is the ``<mujoco>`` model,
  * every body 'kinematics.bodies()' finds is a ``<body>``, nested under the
    body it hangs off at the pose its joint puts it in at the starting values --
    the configuration the assembly shows -- and under the ``<worldbody>`` at its
    world pose when nothing in the tree holds it,
  * every part welded into a body contributes a ``<geom type="mesh">`` at its
    placement in the body, referencing a mesh written next to the model file,
  * every joint is written as MuJoCo's own joints, by what 'kinematics.classify'
    says it is (see 'write_joint'): a hinge or a slide per degree of freedom, a
    ``ball`` for three turns through a point, and an ``<equality>`` for what
    one freedom's value follows -- and a floating joint's child, which nothing
    holds, as a free body in the ``<worldbody>`` (see 'write_floating'),
  * and, with ``names_file``, every joint's *map* goes into the sidecar beside
    the model: which PartCAD joint and parameter each MJCF joint is, in what
    units, from what start ('kinematics.joint_map()'), which is how a run
    reports every joint under PartCAD's name and in PartCAD's terms however it
    had to be written (see 'write_names').

Two things this exporter decides that the file does not say, because a static
arrangement does not say them and a *simulation* needs both:

  * **Whether what is free may move.** A body nothing holds -- a scene element
    not marked ``fixed``, the root of an assembly placed in one -- is free, and
    MuJoCo writes that as a ``<freejoint>``. ``static: true`` (the default, for
    a file somebody opens to look at) welds every such body to the world
    instead, so that a scene stays where it was drawn; the joints *inside* an
    assembly are written either way, so a pendulum exported on its own still
    swings on its hinge. ``static: false`` is what the simulation asks for.
  * **What it stands on.** ``ground_plane`` and ``light`` are not part of what
    the scene says; they are what makes the model usable, the same way the world
    exporter's ``sun`` and ``ground_plane`` are. The plane is a ``<geom>`` of
    the ``<worldbody>`` itself, so it is static whatever ``static`` says.

``flatten`` is no longer an option, and is reported when it is passed: it made
every node holding geometry a free body of its own, which the body rule now
says of every element of a scene and of nothing inside an assembly.

What the scene says about its *world* it says to this exporter in
``request["world"]``, when it says anything (see 'wrappers/wrapper_export.py' in
PartCAD itself): the gravity, in m/s^2, and the fluid it is filled with, as the
material's density (kg/m^3) and dynamic viscosity (Pa*s). They go into
``<option>`` as they arrive -- ``gravity``, ``density`` and ``viscosity`` are
in those units in MJCF too -- which turns on MuJoCo's passive fluid model, and
the fluid's density also becomes each body's buoyancy (see 'write_buoyancy'). A
scene that states neither gets exactly the ``<option>`` it always got. See the
note above 'FLUID_INTEGRATOR' for what MuJoCo's model of a fluid is and is not.

Two conventions are shared with the other two exporters and are what makes the
round trip close: meshes are written in millimetres -- the unit PartCAD uses
everywhere -- and referenced with ``scale="0.001 0.001 0.001"``, which is how
MJCF says "these coordinates are millimetres"; and geometry is written in its
own frame and placed by the element that holds it, so a shape that appears more
than once is written once and referenced by every geom that uses it.

What a body weighs is what it is made of, and this exporter works none of it
out. PartCAD hands every part over with its mass, centre of mass and inertia
already resolved -- stated, or derived from its solid at the density of what it
is made of (see 'partcad.physics' in PartCAD) -- and they go on the body's
``<inertial>``, so a model MuJoCo runs weighs exactly what ``pc info`` says and
what the URDF and SDFormat exports of the same object weigh. A body of several
shapes is added up by PartCAD's own 'mass_properties', the one copy of that
arithmetic all three exporters share. Only a body PartCAD could not weigh -- a
mesh with no solid in it -- is left to MuJoCo, which integrates the mesh at the
density PartCAD resolved for the part.

Note that MuJoCo reads **binary** STL only, which is why ``ascii`` defaults to
false here and an ``ascii: true`` is reported rather than quietly written.
"""

import math
import os
import sys
from xml.etree import ElementTree

# Pinned before anything that may pull OCP - see the note in ocp_serialize.
import pyexpat  # noqa: F401

# 'mujoco_common' is this package's, and sits beside this file. The other four
# are PartCAD's own and are the sandbox contract every implementation is written
# against: 'ocp_serialize' is the shape and assembly envelope format this is
# handed, 'urdf_common' is the pose arithmetic they are stated in,
# 'mass_properties' is how several shapes' inertias add up into one body's, and
# 'kinematics' is which nodes are bodies and how the joints between them move. A
# sandbox runs a wrapper out of PartCAD's 'wrappers/' directory, so all four are
# already on sys.path by the time this is imported.
sys.path.append(os.path.dirname(__file__))
import kinematics  # noqa: E402
import mass_properties  # noqa: E402
import mujoco_common  # noqa: E402
import naming  # noqa: E402
import ocp_serialize  # noqa: E402
import urdf_common  # noqa: E402

# Metres per millimetre, for the mesh ``scale``: the meshes are written in
# millimetres and MJCF reads mesh coordinates as metres after scaling.
MESH_SCALE = 1.0 / urdf_common.MM_PER_M

# MuJoCo's own gravity, written when nothing states another: the scene, the
# 'gravity' option of this export, and a 'simulate:' that passes one to the
# simulation (which applies it over the file) all win over it, in the reverse of
# that order. Earth's, along -Z, which is what every scene meant before a scene
# could say anything else -- and spelled out rather than left out of the file, so
# that a model opened on its own says which way is down.
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

# What MuJoCo makes of a fluid, so that nobody has to find out from a result.
#
# Setting ``<option density viscosity>`` turns on MuJoCo's *passive fluid
# model*, and the default flavour of it is the inertia-box one: each body is
# taken to be the box with its mass and inertia, and the fluid pushes back on it
# with a drag quadratic in its velocity (from the density) and a Stokes
# resistance linear in it (from the viscosity), and with the matching torques.
# That is all. In particular:
#
#   * **No buoyancy.** MuJoCo's fluid model is drag and nothing else, so a block
#     of foam sinks in its water as steadily as a block of lead, only slower. So
#     this exporter adds the buoyancy itself, through MuJoCo's own ``gravcomp``:
#     an upward force on a body's centre of mass of that fraction of its weight,
#     which MuJoCo documents as "a buoyancy effect" when above one. The fraction
#     is the mass of the fluid the body displaces over the body's own mass --
#     Archimedes, scaled by whatever gravity the run ends up with, so a
#     'gravity' override moves the buoyancy with it. Neither is worked out here:
#     the body's mass is the one PartCAD resolved and 'mass_properties.of_body()'
#     put on its '<inertial>', and what it displaces is the volume PartCAD
#     measured its solids to enclose, added up by 'mass_properties.volume_of()'.
#   * **No surface.** The fluid fills the whole world. A body lighter than it
#     rises for ever, at the speed its drag allows, rather than coming to float
#     at a waterline: there is no waterline. A scene with a surface is a
#     different model, and Gazebo's graded buoyancy is the nearest thing to one.
#   * **The centre of buoyancy, but only in a run.** ``gravcomp`` pushes where
#     gravity pulls, at the centre of mass, which for a body of one material is
#     also its centre of volume and so is right. A body whose mass is not
#     centred where its volume is -- a hull with a lead keel, a float that
#     states a low 'centerOfMass' -- would get no righting moment from that,
#     and MJCF has no way to say where else a force acts. So the centre of
#     buoyancy of every buoyed body is written into the model as a
#     ``<custom><numeric>`` (see CENTRE_OF_BUOYANCY_PREFIX), and the simulation
#     beside this applies the moment the lift has about the centre of mass,
#     every step, through ``xfrc_applied``. Opened anywhere else, the model
#     floats but does not right itself.
#   * **Displaced volume is the solid's.** A sealed hollow part is buoyed by the
#     material it is made of, as if flooded. A float is drawn as the solid it
#     displaces and states its own mass.
#   * **No added mass, lift or Magnus effect.** Those are MuJoCo's other,
#     per-geom model (``fluidshape="ellipsoid"``), built for insect flight and
#     tuned per geom with five coefficients nothing in a PartCAD scene states.
#     It is not written: a model that needs it is a model somebody tunes by hand.
#
# MuJoCo recommends an implicit integrator wherever these velocity-dependent
# forces act, so a model with a fluid in it is written with
# ``integrator="implicitfast"``. One with none keeps MuJoCo's default, and every
# existing model the bytes it had.
FLUID_INTEGRATOR = "implicitfast"

# The name of the '<custom><numeric>' each buoyed body's centre of buoyancy is
# written under, followed by the body's name: three numbers, metres, in the
# body's own frame. 'simulate_mujoco.CENTRE_OF_BUOYANCY_PREFIX' is its twin and
# reads it back; the two scripts run in different sandboxes and cannot import
# each other, and a test keeps them the same.
CENTRE_OF_BUOYANCY_PREFIX = "partcad:centre_of_buoyancy:"

# How MuJoCo resolves friction, written into every model unless an export
# says otherwise ('cone', 'impratio', 'noslip_iterations'; null leaves
# MuJoCo's own).
#
# MuJoCo's contacts are soft: a contact under a steady sideways load slips at a
# small steady rate even when the load is well inside the friction cone, and
# with MuJoCo's defaults -- pyramidal cones, an impratio of one, no NoSlip pass
# -- that rate is not small. Two 20 mm aluminium cubes (mu 1.05) stacked in a
# world tilted by 6 degrees, where tan(6 deg) = 0.105 says nothing should move,
# slid 33 mm in ten seconds and the top one fell off. Elliptic cones and an
# impratio of ten -- what MuJoCo's own documentation recommends against slow
# slippage -- bring that to about a millimetre, and three NoSlip iterations
# (its stronger remedy) to under half a millimetre at 15 degrees, while a top
# block whose friction is below tan(15 deg) still slides off within a second.
# Without these, whether a stack "stands" is a question about the solver rather
# than about the material it is made of.
CONTACT_DEFAULTS = {"cone": "elliptic", "impratio": 10, "noslip_iterations": 3}

# What an MJCF name may be, declared here because it is MJCF's (see PartCAD's
# 'wrappers/naming.py'). MuJoCo's compiler requires names to be unique among
# the elements of one type -- a body and a geom may share one, two bodies may not
# -- compares them exactly, and accepts any characters at any length: '//p/x:y z'
# compiles, and so does 'Cube' beside 'cube' (checked against MuJoCo 3.16; the
# XML reference says the same of every 'name' attribute). One name is taken: the
# '<worldbody>' is the body called 'world', so a body of that name is "repeated
# name 'world' in body". Hence two rule sets, and one transcoder per element type
# in 'process'. What is left to the transcoder is the XML itself: no control
# characters, which an attribute does not keep.
MJCF_NAMES = naming.Rules("MJCF")
MJCF_BODY_NAMES = naming.Rules("MJCF", reserved=("world",))

# PartCAD property -> the ``<geom>`` attribute that states it, and how to write
# it. MJCF states friction as one attribute holding three coefficients, so the
# sliding one -- which is what PartCAD's 'friction' is -- is filled in beside
# MuJoCo's own defaults for the other two.
GEOM_PHYSICS = {
    "friction": ("friction", lambda v: mujoco_common.format_numbers([float(v), 0.005, 0.0001], 6)),
    "restitution": ("solref", lambda v: _solref_for_restitution(float(v))),
}

# Every part property this exporter has an MJCF spelling for. A 'physics'
# property outside this set is one PartCAD supports and MJCF does not: it is
# reported through the response and logged at info level, which is the mirror
# image of the reader reporting what it cannot keep. 'density' is the geom's own
# attribute of that name, written where MuJoCo has to weigh a mesh itself, and
# otherwise what the '<inertial>' PartCAD worked out was worked out from.
#
# 'volume' and 'centerOfVolume' are not properties MJCF states either, and are
# not ones a part states: they are what PartCAD measured the solid to enclose
# and where, and they go into the model as what the body displaces of the
# scene's fluid and where it is lifted from (see 'write_buoyancy').
MJCF_STATED = frozenset(
    (
        "mass",
        "centerOfMass",
        "inertiaOrientation",
        "inertia",
        "density",
        "volume",
        "centerOfVolume",
        "friction",
        "restitution",
    )
)

# A joint's physics, as PartCAD states it on an interface, a mating or a
# connection -> the MJCF ``<joint>`` attribute that states it. Every one is per
# unit of the joint's own motion in SI -- N*m*s/rad and N*s/m for 'damping',
# N*m and N for 'friction', N*m/rad and N/m for 'springStiffness' -- and so is
# MuJoCo's: it integrates a hinge in radians and a slide in metres whatever the
# ``<compiler>`` says the file is written in. So they are written as they
# arrive. 'springReference' is a position, degrees or millimetres, and is
# converted as a joint's 'ref' is (see 'joint_value').
JOINT_PHYSICS = {
    "damping": "damping",
    "friction": "frictionloss",
    "springStiffness": "stiffness",
}
SPRING_REFERENCE = "springReference"

# The joint physics PartCAD holds that a MuJoCo model has nowhere to put, which
# are reported rather than written approximately:
#
#   * 'maxEffort' bounds what drives a joint, and MJCF says it as a joint's
#     ``actuatorfrcrange``: a clamp on the force the model's *actuators* apply
#     along it. A model PartCAD writes has no actuators, so there is nothing to
#     clamp, and a range written there would say nothing about the joint. It
#     belongs with the actuator that drives the joint (roadmap step 14).
#   * 'maxVelocity': MuJoCo limits no joint's velocity.
#   * 'stopCfm', 'stopErp', 'fudgeFactor', 'implicitSpringDamper' and
#     'provideFeedback' are another engine's solver settings (ODE's, through
#     URDF's <gazebo> extension), with no MuJoCo meaning.
#   * 'softLimits' are a controller's, and a model has none.
#
# MJCF's ``armature`` -- the inertia a motor's rotor adds to a joint -- is not in
# PartCAD's joint physics, so nothing is written for it: it is a property of a
# motor, and arrives with one (roadmap step 14).

# The open end of an MJCF ``range``, where PartCAD bounds a freedom on one side
# only: a door that stops at 90 degrees one way and swings freely the other.
# MJCF states a range as a pair, and MuJoCo's XML reader takes ``inf`` and
# ``-inf`` for either end of it ('ParseInfOrNan' in its 'xml_util.cc', there
# since 3.2): its compiler asks only that the lower end be below the upper, and
# the limit on the open side measures an infinite distance, so it never acts
# (docs/design/simulation.md in PartCAD, section 8.13.6). It replaces the million
# degrees -- or kilometre -- this exporter wrote there before, which was a number
# nobody chose standing for "no bound". A freedom bounded on neither side is
# ``limited="false"`` and has no range at all.
OPEN = math.inf

# Between a PartCAD joint's name and the freedom an MJCF joint is of, where one
# PartCAD joint is written as several MJCF joints: 'nut-bolt.turnZ' and
# 'nut-bolt.moveZ'. A dot, which every format PartCAD's simulations read allows.
# These are the names *in the file*, for whoever opens it: a run reads every
# MJCF joint of one PartCAD joint back as that joint, through the joint map in
# the sidecar (see 'write_names'), and reports it under the PartCAD name alone.
FREEDOM_SEPARATOR = "."

# The '<keyframe><key>' a model starts from when one of its joints cannot be
# written where the assembly shows it: a ball bounded by a cone that starts away
# from its zero (see 'write_ball'). 'simulate_mujoco.START_KEY' is its twin and
# resets a run to it ('mj_resetDataKeyframe'); the two scripts run in different
# sandboxes and cannot import each other, and a test keeps them the same.
START_KEY = "partcad:start"


def _solref_for_restitution(restitution):
    """MuJoCo's ``solref`` for a coefficient of restitution, or None.

    MuJoCo states contact softness as a (time constant, damping ratio) pair
    rather than as a restitution, and a damping ratio below one is what makes a
    contact bounce. The mapping is not exact -- nothing in MuJoCo is a
    restitution -- so it is only written for a part that states one, and only
    as the damping ratio the value most nearly means.
    """
    restitution = max(0.0, min(1.0, restitution))
    if restitution <= 0.0:
        return None
    return mujoco_common.format_numbers([0.02, max(0.01, 1.0 - restitution)], 6)


def model_name_of(name, fallback):
    """The MJCF model's name for a PartCAD object's name.

    A parameterized object's name also carries its parameter values, after a
    ';'. Those are dropped rather than spelled out: `pc sim` runs a scene whose
    subject *is* a parameter, so keeping them would make the model's name a
    transcription of the whole declaration -- and a name is read, not parsed.
    Which instance it is is what the run is about and is reported beside the
    file, not smuggled into it. What is left is the scene's own name, 'world'
    included: MJCF reserves that for a body, not for a model.
    """
    return naming.transcode(MJCF_NAMES, str(name or "").partition(";")[0], fallback)


# Where PartCAD puts a node's own geometry, decoded, on the tree it hands over
# as 'shape_tree'. Spelled out rather than read off 'ocp_serialize', which only
# a PartCAD that hands 'shape_tree' defines it in.
KEY_SHAPE = "shape"


def tree_of(request):
    """The tree to write, as PartCAD handed it over.

    'shape_tree' from a PartCAD that hands every exporter the tree beside the
    geometry; 'wrapped' from one released before it did, which hands the tree
    there instead to an exporter that declares 'decode: false'.
    """
    tree = request.get("shape_tree")
    return tree if tree is not None else request.get("wrapped")


def node_geometry(node):
    """The node's own shape as a live TopoDS_Shape, in its own frame, or None.

    Decoded already on a node of 'shape_tree'; decoded here on one of the raw
    tree an older PartCAD hands over. The placement carried by the node is
    deliberately left off either way: it becomes the pose of the element above
    the geometry, not part of the geometry.
    """
    if not ocp_serialize.is_shape_object(node):
        return None
    if node.get(KEY_SHAPE) is not None:
        return node[KEY_SHAPE]
    without_placement = {key: value for key, value in node.items() if key != ocp_serialize.KEY_LOCATION}
    return ocp_serialize.decode_shape(without_placement)


# The most triangles MuJoCo reads out of one STL file. Its decoder refuses a file
# with more ("number of faces should be between 1 and 200000"), and the model with
# it -- which a finely detailed part (a worm's thread, a knurled knob) reaches at
# the default tolerance.
MAX_STL_FACES = 200000

# How many times a mesh MuJoCo could not read is triangulated again, each time
# twice as coarse, before the export gives up on it.
COARSER_ATTEMPTS = 8


def triangle_count(shape):
    """How many triangles the triangulation 'shape' carries now has."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        triangulation = BRep_Tool.Triangulation_s(TopoDS.Face_s(explorer.Current()), TopLoc_Location())
        if triangulation is not None:
            count += triangulation.NbTriangles()
        explorer.Next()
    return count


def write_mesh(shape, path, options):
    """Triangulate 'shape' and write it out as an STL file; return the (tolerance, triangles) it was written at.

    At the options' tolerance, unless that makes more triangles than MuJoCo
    reads from one file ('MAX_STL_FACES'): then twice as coarse, and again,
    until it does not -- a model with one mesh a little coarser than asked is
    one MuJoCo can load, and one with a mesh it refuses is not. The caller
    says so when it happens.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.StlAPI import StlAPI_Writer

    tolerance, angular = options["tolerance"], options["angularTolerance"]
    previous = None
    for attempt in range(COARSER_ATTEMPTS + 1):
        BRepMesh_IncrementalMesh(
            shape,
            theLinDeflection=tolerance,
            isRelative=True,
            theAngDeflection=angular,
            isInParallel=True,
        )
        triangles = triangle_count(shape)
        if triangles <= MAX_STL_FACES:
            break
        if triangles == previous:
            # Nothing to triangulate coarser: a mesh already (an imported STL
            # keeps its triangles, whatever the tolerance), or that many flat
            # faces, each a triangle or two at any tolerance.
            raise Exception(
                "%s has %d triangles, more than the %d MuJoCo reads from one mesh, and no coarser tolerance makes "
                "fewer: it is a mesh already, or that many flat faces. Simplify it before it is simulated"
                % (os.path.basename(path), triangles, MAX_STL_FACES)
            )
        if attempt == COARSER_ATTEMPTS:
            raise Exception(
                "%s has %d triangles at a tolerance of %g, more than the %d MuJoCo reads from one mesh: "
                "give the export a coarser 'tolerance'" % (os.path.basename(path), triangles, tolerance, MAX_STL_FACES)
            )
        # A triangulation is kept on the shape and only ever refined, so the
        # coarser one is made from nothing.
        BRepTools.Clean_s(shape)
        previous = triangles
        tolerance, angular = tolerance * 2.0, min(angular * 2.0, 1.0)
    writer = StlAPI_Writer()
    writer.ASCIIMode = options["ascii"]
    if not writer.Write(shape, path) or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise Exception("Failed to write the mesh file: %s" % path)
    return tolerance, triangles


def write_buoyancy(body, inertial, parts, state):
    """Buoy 'body' up with the weight of the fluid it displaces, as MuJoCo's ``gravcomp``.

    'inertial' is the mass properties the body was just written with -- what
    'mass_properties.of_body()' made of what PartCAD resolved -- and 'parts' the
    same (physics, placement) pairs it was made from, which carry the volume
    PartCAD measured each solid to enclose and its centroid. The fraction is
    the mass of the fluid displaced over the body's own:
    'mass_properties.mass_of()' of the body's volume at the fluid's density,
    over the body's mass. Where that lift acts is the body's centre of volume,
    'mass_properties.displacement_of()', which is recorded for
    'write_centres_of_buoyancy()'. Nothing is measured or weighed here, so the
    buoyancy and the '<inertial>' are of one body and cannot disagree about it.

    Nothing is written for a scene with no fluid. A body PartCAD could not weigh
    (no '<inertial>', so MuJoCo weighs the mesh itself) or whose volume is not
    known -- an open mesh, a shell -- is given none, and that is reported: a
    body buoyed by a guess floats or sinks for no reason anybody could find.
    """
    fluid = state["fluid_density"]
    if not fluid:
        return
    displacement = mass_properties.displacement_of(parts)
    volume = displacement["volume"] if displacement else mass_properties.volume_of(parts)
    mass = inertial.get("mass") if inertial else None
    if volume is None or not mass:
        state["warnings"].append(
            "%s has no %s PartCAD could resolve, so it is given no buoyancy in the fluid the scene is filled with"
            % (body.get("name"), "mass" if volume is not None else "enclosed volume")
        )
        return
    displaced = mass_properties.mass_of(volume, fluid)
    body.set("gravcomp", mujoco_common.format_numbers([displaced / float(mass)], 6))
    if displacement is None:
        state["warnings"].append(
            "%s has no centre of volume PartCAD could resolve, so it is buoyed at its centre of mass and is not "
            "righted" % body.get("name")
        )
        return
    # Where the lift acts, in the body's frame and in metres, for the
    # simulation to apply the moment about the centre of mass from.
    centre = [value / urdf_common.MM_PER_M for value in displacement[mass_properties.CENTER_OF_VOLUME_KEY]]
    state["centres_of_buoyancy"].append((body.get("name"), centre))


def write_centres_of_buoyancy(mujoco, state):
    """Write each buoyed body's centre of buoyancy as a ``<custom><numeric>``.

    MJCF has a place for arbitrary numbers a program reads back -- custom
    numerics, which MuJoCo loads and ignores -- and none for "apply this force
    here", so this is where the simulation finds the point. Nothing is written
    for a model with nothing buoyed in it, so every other model is the file it
    always was.
    """
    if not state["centres_of_buoyancy"]:
        return
    custom = ElementTree.SubElement(mujoco, "custom")
    for name, centre in state["centres_of_buoyancy"]:
        numeric = ElementTree.SubElement(custom, "numeric")
        numeric.set("name", CENTRE_OF_BUOYANCY_PREFIX + name)
        numeric.set("data", mujoco_common.format_numbers(centre))


def mesh_asset(shape, node, body_name, state):
    """Write 'shape' out as a mesh (or reuse one) and return the asset's name.

    An identical shape used twice - a repeated fastener, a row of blocks -
    shares one mesh file and one ``<asset><mesh>``, the way a model written by
    hand would.
    """
    key = node.get(ocp_serialize.KEY_BREP)
    asset_name = state["meshes"].get(key)
    if asset_name is not None:
        return asset_name

    asset_name = state["mesh_names"].derive(node.get("label") or body_name, "mesh")
    tolerance, _triangles = write_mesh(shape, os.path.join(state["mesh_dir"], "%s.stl" % asset_name), state["options"])
    if tolerance != state["options"]["tolerance"]:
        state["warnings"].append(
            "'%s' has more triangles than MuJoCo reads from one mesh at a tolerance of %g, so it is written at %g"
            % (node.get("label") or body_name, state["options"]["tolerance"], tolerance)
        )
    mesh = ElementTree.SubElement(state["asset"], "mesh")
    mesh.set("name", asset_name)
    mesh.set("file", "%s/%s.stl" % (state["mesh_dir_name"], asset_name))
    mesh.set("scale", mujoco_common.format_numbers((MESH_SCALE,) * 3, 6))
    state["meshes"][key] = asset_name
    return asset_name


def carried_inertial(physics):
    """The ``<inertial>`` values of a body, or None if it has no mass.

    What PartCAD resolved -- a mass the part states, or the one its solid comes
    to at its density, with the centre and the inertia that go with it -- so
    nothing is computed here. The tensor is written to twelve significant
    digits, and for anything with a solid in it is the integral of a real
    distribution of mass, which is positive definite: the rounding that once
    made computing one here risky is far below what MuJoCo's check can see.
    """
    if not physics or "mass" not in physics:
        return None
    values = {"mass": float(physics["mass"])}
    if "centerOfMass" in physics:
        values["centerOfMass"] = [float(v) for v in physics["centerOfMass"]]
    if "inertiaOrientation" in physics:
        values["inertiaOrientation"] = [float(v) for v in physics["inertiaOrientation"]]
    inertia = physics.get("inertia")
    if isinstance(inertia, dict) and any(abs(float(inertia.get(key, 0.0))) > 0 for key in ("ixx", "iyy", "izz")):
        values["inertia"] = {key: float(inertia.get(key, 0.0)) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    return values


def write_inertial(body, values):
    """Write one body's ``<inertial>`` from the values a part stated."""
    if values is None:
        return
    inertial = ElementTree.SubElement(body, "inertial")
    centre = values.get("centerOfMass") or (0.0, 0.0, 0.0)
    inertial.set("pos", mujoco_common.format_numbers([v / urdf_common.MM_PER_M for v in centre]))
    inertial.set("mass", mujoco_common.format_numbers([values["mass"]]))
    orientation = values.get("inertiaOrientation")
    if orientation:
        rotation = urdf_common.rpy_to_quat([math.radians(v) for v in orientation])
        inertial.set("quat", mujoco_common.format_numbers(rotation))
    inertia = values.get("inertia")
    if inertia:
        # MJCF orders 'fullinertia' as ixx iyy izz ixy ixz iyz.
        inertial.set(
            "fullinertia",
            mujoco_common.format_numbers(
                [inertia["ixx"], inertia["iyy"], inertia["izz"], inertia["ixy"], inertia["ixz"], inertia["iyz"]]
            ),
        )
    else:
        # MuJoCo needs one of the two, and a body that states a mass and no
        # tensor is a point mass as far as the file is concerned. A tiny
        # diagonal keeps it a body rather than making it singular.
        inertial.set("diaginertia", mujoco_common.format_numbers([1e-6, 1e-6, 1e-6]))


def _positive(value):
    """'value' as a positive finite float, or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def physics_of(node, state):
    """What PartCAD says about the physics of the object behind one node of the tree.

    Looked up by 'kinematics.properties_of', not by the node's full name: an
    assembly or a scene whose full name a part shares -- the assembly 'nut'
    that holds the part 'nut' -- has an entry of its own under its kind
    ('kinematics.properties_key'), and looked up by name it got the part's
    contact settings. A part is under its full name, as it always was.
    """
    return (kinematics.properties_of(state["properties"], node) or {}).get("physics") or {}


def emit_geom(body, part, pose, asset_name, physics, weighed, state, index):
    """Add one ``<geom>`` to a body, with what its part says about itself.

    'pose' is where the part sits in the body. 'physics' is what decides how
    the geom touches things -- its part's own, over what the body's own node
    states -- and the part's properties say its colour and, where the body has
    no ``<inertial>``, its density.

    'weighed' says whether the body has an ``<inertial>``. One that has not is
    one PartCAD could not weigh -- a mesh with no solid in it -- and MuJoCo
    weighs it instead, from the mesh, at the density PartCAD resolved for the
    part: its own, its material's, or the export's. PartCAD resolves one for
    every part it exports, so a geom without a usable one is reported rather
    than left to MuJoCo's default of water's, which nobody chose.
    """
    geom = ElementTree.SubElement(body, "geom")
    geom.set("name", state["geom_names"].derive("%s_geom_%d" % (body.get("name"), index), "geom"))
    geom.set("type", "mesh")
    geom.set("mesh", asset_name)
    if not urdf_common.is_identity(pose):
        geom.set("pos", mujoco_common.format_pos(pose))
        geom.set("quat", mujoco_common.format_quat(pose))

    properties = state["properties"].get(part.name) or {}
    rgba = mujoco_common.color_rgba(properties.get("color"))
    if rgba is not None:
        geom.set("rgba", rgba)

    if not weighed:
        stated = (properties.get("physics") or {}).get("density")
        density = _positive(stated)
        if density is not None:
            geom.set("density", mujoco_common.format_numbers([density], 6))
        else:
            state["warnings"].append(
                "The geom '%s' has no mass PartCAD could work out and no usable density (%r); "
                "MuJoCo will weigh its mesh at its own default of 1000 kg/m^3" % (geom.get("name"), stated)
            )

    for name, (attribute, render) in GEOM_PHYSICS.items():
        if name not in physics:
            continue
        value = render(physics[name])
        if value is not None:
            geom.set(attribute, value)
    return geom


def shape_pairs(body):
    """A body's shapes as the (physics, packed placement) pairs 'mass_properties' reads."""
    return [(part.physics, kinematics.packed_of(part.placement)) for part in body.parts]


def emit_body(parent, body, pose, state):
    """Add one ``<body>`` for a body 'kinematics' found, with a geom per part welded into it.

    Its mass, centre of mass and inertia are 'body.physics' -- its parts added
    up by 'mass_properties.of_body()', each where the body holds it, or what an
    assembly welded whole into it states it weighs -- so nothing is weighed
    here. Returns the element and what 'emit_geoms' writes into it, or None
    for a body with nothing in it and nothing under it, which goes back out.
    """
    body_name = state["body_names"].claim(body.name, "body", "body")
    element = ElementTree.SubElement(parent, "body")
    element.set("name", body_name)
    if not urdf_common.is_identity(pose):
        element.set("pos", mujoco_common.format_pos(pose))
        element.set("quat", mujoco_common.format_quat(pose))

    inertial = carried_inertial(body.physics)
    if inertial is not None:
        write_inertial(element, inertial)

    # What the body's own node says decides how its parts touch things where a
    # part says nothing itself: a link of several shapes read from a robot
    # description states its friction once, on the link.
    own = physics_of(body.node, state)
    shapes = [(index, part, node_geometry(part.node)) for index, part in enumerate(body.parts)]
    shapes = [entry for entry in shapes if entry[2] is not None]

    if not shapes and not body.children:
        # A frame with nothing in it and nothing under it. MuJoCo accepts an
        # empty body, but it is noise in the model and in what a simulation
        # reports, so it goes back out.
        parent.remove(element)
        state["body_names"].release(body_name)
        if body.joint is not None:
            state["warnings"].append(
                "'%s' holds no geometry, so its joint '%s' is not written" % (body.name, body.joint.name)
            )
        return None

    # Which parts this body is, for the sidecar: what lets a run say which
    # body a part is in and where, and so read a body's loads as a part's. A
    # PartCAD from before 'parts_entry()' cannot say, and its runs report the
    # loads without 'parts'.
    parts_entry = getattr(kinematics, "parts_entry", None)
    if parts_entry is not None:
        state["body_parts"][body.name] = parts_entry(body)
    state["unsupported"].update(set(own) - MJCF_STATED)
    for _, part, _shape in shapes:
        state["unsupported"].update(set(part.physics) - MJCF_STATED)
    return element, own, shapes, inertial


def emit_geoms(element, own, shapes, inertial, body, state):
    """Write a body's geoms, and the buoyancy of what they displace."""
    for index, part, shape in shapes:
        asset_name = mesh_asset(shape, part.node, element.get("name"), state)
        physics = dict(own)
        physics.update(part.physics)
        emit_geom(element, part, part.placement, asset_name, physics, inertial is not None, state, index)
    if shapes:
        write_buoyancy(element, inertial, shape_pairs(body), state)


#
# Joints
#


def joint_value(kind, value):
    """A freedom's position in the units this file states it in: degrees for a turn, metres for a move.

    The ``<compiler angle="degree">`` this exporter writes makes a hinge's
    'ref', 'range' and 'springref' degrees -- PartCAD's own unit, so a turn is
    written as it is declared. A move is millimetres in PartCAD and metres in
    MJCF.
    """
    return float(value) if kind == kinematics.TURN else float(value) / urdf_common.MM_PER_M


def si_scale(kind) -> float:
    """What one PartCAD unit of a freedom is in MuJoCo's own: radians per degree, metres per millimetre.

    MuJoCo computes in those whatever the ``<compiler>`` says, and an
    ``<equality joint>``'s 'polycoef' is read as it is written, in them.
    """
    return math.pi / 180.0 if kind == kinematics.TURN else 1.0 / urdf_common.MM_PER_M


def freedom_label(joint, freedom) -> str:
    """What an MJCF joint of one freedom is called after its PartCAD joint's name: its parameter.

    'kinematics.Joint.parameter_names()', which is what a reading of the joint
    keys that freedom by too -- its parameter, or its kind and index where no
    parameter declared it, told apart within the joint -- so that the file and
    the run say the same of it.
    """
    return joint.parameter_names()[freedom.index]


def joint_name(body) -> str:
    """The name a run reports 'body''s joint under: PartCAD's, or its two bodies' where it has none."""
    joint = body.joint
    parent = body.joint_parent
    return joint.name or "%s-%s" % (body.name, parent.name if isinstance(parent, kinematics.Body) else parent)


def is_keyed(joint) -> bool:
    """Whether 'joint' is a ball bounded by a cone that starts away from its zero (see 'write_ball')."""
    if joint.kind != kinematics.BALL or joint.turn_limit is None:
        return False
    rotation = kinematics.rezeroed(joint).rotation
    # How far the start is turned from the zero, as the sine of half of it:
    # nought, to rounding, for a start of no turn at all (or of whole turns).
    return math.sqrt(sum(float(v) * float(v) for v in rotation[1:])) > 1e-12


def written_location(body):
    """Where 'body' is written in its parent body: the configuration the assembly shows, but for a keyed ball.

    Every joint is written at its starting values -- a hinge and a slide at
    their 'ref', a ball re-zeroed at its start -- except a ball whose cone has
    to stay about PartCAD's zero, which is written at the zero and started by
    the model's keyframe (see 'write_ball').
    """
    joint = body.joint
    if is_keyed(joint):
        return joint.location([0.0] * len(joint.freedoms))
    return joint.location()


def write_joint(element, body, state):
    """Write the joint 'body' hangs off its parent by, as MuJoCo's own joints.

    What 'kinematics.classify' says the joint is decides how (section 8.13.4 of
    PartCAD's simulation design record):

      revolute, prismatic,    one hinge or slide per degree of freedom, in the
      cylindrical, screw,     order declared, in this one body: MuJoCo applies
      universal, planar       a body's joints in the order they are written,
      (either order),         each about its axis through its point in the
      composite               frame the ones before it left, which is the
                              decomposition 'kinematics' gives every joint
                              ('pos' and 'axis' are 'anchor_in_child' and
                              'axis_in_child'); a screw's thread is an
                              '<equality joint>' between its two
      ball                    one 'ball', centred where the three turns meet
                              ('origin_in_child'); see 'write_ball'

    A floating joint is not written here: its child is a body of its own, held
    by nothing, and written free in the '<worldbody>' (see 'write_floating').

    Each hinge and slide starts where the assembly shows it: the body is
    written at the starting values, and the joint's 'ref' is the starting
    value, so MuJoCo's zero is PartCAD's zero and its reading of the joint is
    the value PartCAD gives it. A ball has no 'ref', so it is re-zeroed
    instead: written at its start, MuJoCo's identity *is* the start, and the
    joint map says what to turn a reading by to make it PartCAD's.

    Every MJCF joint written goes into the sidecar with its joint map --
    'kinematics.joint_map()', which PartCAD's joint and parameter it is, in what
    units, from what start -- and a run reads them all back as the one PartCAD
    joint they were written for ('kinematics.reading()'). In the file, a joint
    of one MJCF joint is named exactly what PartCAD named it, and one written as
    several is '<name>.<parameter>' for each. What one freedom follows -- a
    screw's move its turn -- is recorded for 'write_equalities'.
    """
    joint = body.joint
    name = joint_name(body)
    freedoms = joint.freedoms
    written = {}

    if joint.kind == kinematics.BALL:
        write_ball(element, body, name, state)
    else:
        single = len(freedoms) == 1
        bounds = freedom_bounds(joint)
        for freedom in freedoms:
            mjcf_name = name if single else name + FREEDOM_SEPARATOR + freedom_label(joint, freedom)
            written[freedom.index] = _write_freedom(element, joint, freedom, mjcf_name, name, bounds, state)

    for freedom in freedoms:
        if freedom.follows is not None and freedom.index in written and freedom.follows in written:
            leader = freedoms[freedom.follows]
            state["equalities"].append(
                (written[freedom.index], freedom, written[leader.index], leader, freedom.ratio, freedom.offset)
            )

    for key in sorted(set(joint.physics) - set(JOINT_PHYSICS) - {SPRING_REFERENCE}):
        state["unsupported"].add("joint %s" % key)
    if joint.soft_limits:
        state["unsupported"].add("joint softLimits")

    scalar_joints = [(written[index], freedoms[index]) for index in sorted(written)]
    state["joints"].setdefault(joint.name, []).append(scalar_joints)
    if joint.mimic:
        state["mimics"].append((name, joint.mimic, scalar_joints))


def freedom_bounds(joint) -> dict:
    """Each freedom's bounds as written, '{index: (lower, upper)}' in PartCAD's units, None where open.

    A freedom's own, but for a screw's move: a screw is bounded by its turn
    (section 8.13.3 of PartCAD's simulation design record), so a bound stated
    on the move is written as the turn's, through the lead -- the turn at which
    the move reaches it -- and laid over whatever the turn states itself, the
    narrower of the two winning; the move is then written with none
    (section 8.13.6). MuJoCo would hold a bound on the slide as well, but only
    through the equality that couples the two, which it holds softly.
    """
    bounds = {freedom.index: (freedom.lower, freedom.upper) for freedom in joint.freedoms}
    if joint.kind != kinematics.SCREW:
        return bounds
    for freedom in joint.freedoms:
        if freedom.follows is None or not freedom.limited or not freedom.ratio:
            continue
        leader = joint.freedoms[freedom.follows]
        offset = freedom.offset or 0.0
        # move = ratio * turn + offset, so the turn at a bound of the move is
        # (bound - offset) / ratio -- and the move's lower bound is the turn's
        # upper one where the thread runs the other way.
        through = [None if bound is None else (bound - offset) / freedom.ratio for bound in bounds[freedom.index]]
        if freedom.ratio < 0:
            through.reverse()
        lower, upper = bounds[leader.index]
        if through[0] is not None:
            lower = through[0] if lower is None else max(lower, through[0])
        if through[1] is not None:
            upper = through[1] if upper is None else min(upper, through[1])
        bounds[leader.index] = (lower, upper)
        bounds[freedom.index] = (None, None)
    return bounds


def _joint_element(element, state, name, kind, anchor, reported, entry):
    """One MJCF joint, under the name 'name' made an MJCF name, recorded for the sidecar.

    'reported' is the PartCAD joint it is (a freedom of), which a run reports
    it under, and 'entry' its joint map ('kinematics.joint_map()'): both go
    into the sidecar beside where the joint is (see 'write_names').
    """
    internal = state["joint_names"].claim(name, "joint", "joint")
    state["reported_joints"][name] = {"path": [internal], "joint": reported, "map": entry}
    joint = ElementTree.SubElement(element, "joint")
    joint.set("name", internal)
    joint.set("type", kind)
    joint.set("pos", mujoco_common.format_numbers([v / urdf_common.MM_PER_M for v in anchor]))
    return joint


def _write_freedom(element, joint, freedom, name, reported, bounds, state):
    """One hinge or slide, of one freedom of 'joint'. Returns the name it was written under.

    Its map is 'kinematics.joint_map(joint, [freedom], start=False)': the
    joint is not re-zeroed, because its 'ref' already makes MuJoCo's value
    PartCAD's, and a run has only to convert the units.
    """
    mjcf = _joint_element(
        element,
        state,
        name,
        "hinge" if freedom.kind == kinematics.TURN else "slide",
        joint.anchor_in_child(freedom.index),
        reported,
        kinematics.joint_map(joint, [freedom.index], start=False),
    )
    mjcf.set("axis", mujoco_common.format_numbers(joint.axis_in_child(freedom.index)))
    if freedom.value:
        mjcf.set("ref", mujoco_common.format_numbers([joint_value(freedom.kind, freedom.value)]))
    lower, upper = bounds[freedom.index]
    if lower is not None or upper is not None:
        lower = -OPEN if lower is None else lower
        upper = OPEN if upper is None else upper
        mjcf.set("limited", "true")
        mjcf.set(
            "range",
            mujoco_common.format_numbers(
                [joint_value(freedom.kind, min(lower, upper)), joint_value(freedom.kind, max(lower, upper))]
            ),
        )
    else:
        mjcf.set("limited", "false")
    if freedom.follows is None:
        # What moving the joint costs is a cost of moving its own freedoms; one
        # that follows another (a screw's move) is moved by it, and damping it
        # too would charge the same motion twice.
        _write_joint_physics(mjcf, joint, freedom.kind, reported, state)
    return mjcf.get("name")


def write_ball(element, body, name, state):
    """One MJCF 'ball' for a ball joint, at the centre its three turns go through.

    MJCF bounds a ball by one angle, how far it turns from its zero -- which is
    how PartCAD bounds one too ('joint.turn_limit') -- and writes it as the
    ball's ``range="0 a"``. MuJoCo's ball has no starting value: its identity is
    "the initial configuration in which the model is defined", and a bound is
    a cone about that same configuration. So:

      * **A ball with no bound, or one that starts at its zero, is re-zeroed**
        (section 8.13.5 of PartCAD's simulation design record): the body is
        written at its start, MuJoCo's identity is the start, and the map's
        'rotation' is the start R_s, which a reading puts back in front of
        MuJoCo's rotation, J' = J . R_s. A ball at its zero is the same
        either way, so its cone is about PartCAD's zero.
      * **A bounded ball that starts anywhere else is not.** Re-zeroed, its cone
        would be about its start; so the body is written at the ball's zero
        orientation, where the cone belongs, and the start goes into the
        model's keyframe instead (see 'write_keyframe'), which the simulation
        resets to before it reads anything. Its map's start is the identity.
        This is reported, because MuJoCo opens a file at its own initial
        configuration and not at a keyframe: the ball is shown at its zero.

    MuJoCo states a ball's rotation in the child body's frame, which the map
    records ('kinematics.CHILD_FRAME') so that a reading is in the ball's own.
    """
    joint = body.joint
    keyed = is_keyed(joint)
    values = [0.0] * len(joint.freedoms) if keyed else None
    centre = joint.origin_in_child(values)[1]
    ball = _joint_element(
        element,
        state,
        name,
        "ball",
        centre,
        name,
        kinematics.joint_map(joint, start=not keyed, frame=kinematics.CHILD_FRAME),
    )
    if joint.turn_limit is not None:
        ball.set("limited", "true")
        ball.set("range", mujoco_common.format_numbers([0.0, joint.turn_limit]))
    else:
        ball.set("limited", "false")
    _write_joint_physics(ball, joint, kinematics.TURN, name, state, ball=True)
    if not keyed:
        return
    # The start in MuJoCo's terms: its rotation of the child body as written,
    # in the child body's frame, which is R_s seen from that frame.
    start = kinematics.rezeroed(joint).rotation
    tail = joint.classification.tail[0]
    engine = urdf_common.quat_mul(urdf_common.quat_mul((tail[0], -tail[1], -tail[2], -tail[3]), start), tail)
    state["keyed"][ball.get("name")] = urdf_common.normalize(engine)
    turned = math.degrees(2.0 * math.acos(min(1.0, abs(float(start[0])))))
    state["warnings"].append(
        "The joint '%s' is a ball bounded at %g degrees that starts %.6g degrees from its zero: MJCF bounds a ball "
        "about the orientation its body is written in, so the body is written at the ball's zero and its start is "
        "the keyframe '%s', which a run starts from. MuJoCo opens the file at the ball's zero"
        % (name, joint.turn_limit, turned, START_KEY)
    )


def write_floating(element, body, parent, state):
    """A floating joint's child, which is written free: a ``<freejoint>``, under PartCAD's name for the joint.

    A floating joint holds nothing. 'kinematics' makes its child a body of its
    own, with no parent (section 8.13.2 of PartCAD's simulation design record),
    which MJCF writes as a body directly in the '<worldbody>' with a
    '<freejoint>' -- the only place MuJoCo allows one. Nested under what it was
    connected to, as three slides and a ball, a loose part would have passed
    through it: MuJoCo filters a child's contacts with its parent, and this
    exporter excludes the two bodies of every joint. So nothing is excluded
    here, and a block dropped on its base comes to rest on it.

    It is written whatever 'static' says, as every joint an assembly declares
    is: 'static' welds what *nothing* in the tree holds, and this body is loose
    because the assembly says so. It is reported as the joint it was declared
    as, read against the body it was connected to ('parent', the element of
    'body.joint_parent', or None where that body was not written): its map is
    'kinematics.joint_map(joint)', which a run reads from the two bodies' poses.
    A joint's physics is between the two bodies it joins, and a free joint
    moves against the world, so none of it is written.
    """
    joint = body.joint
    name = joint_name(body)
    internal = state["joint_names"].claim(name, "joint", "joint")
    free = ElementTree.SubElement(element, "freejoint")
    free.set("name", internal)
    if parent is None:
        state["warnings"].append(
            "'%s' is floating against '%s', which holds no geometry and is not written: it is a free body, and "
            "its joint '%s' is not reported" % (body.name, body.joint_parent.name, name)
        )
    else:
        state["reported_joints"][name] = {"path": [internal], "joint": name, "map": kinematics.joint_map(joint)}
    stated = sorted(set(joint.physics) & (set(JOINT_PHYSICS) | {SPRING_REFERENCE}))
    if stated or joint.mimic:
        state["warnings"].append(
            "The joint '%s' is floating, which MJCF writes as a free body moving against the world rather than "
            "against '%s': its %s %s not written"
            % (
                name,
                body.joint_parent.name,
                ", ".join(stated + (["mimic"] if joint.mimic else [])),
                "is" if len(stated) + bool(joint.mimic) == 1 else "are",
            )
        )
    for key in sorted(set(joint.physics) - set(JOINT_PHYSICS) - {SPRING_REFERENCE}):
        state["unsupported"].add("joint %s" % key)


def _write_joint_physics(mjcf, joint, kind, name, state, ball=False):
    physics = joint.physics
    for key, attribute in JOINT_PHYSICS.items():
        value = physics.get(key)
        if value is None:
            continue
        try:
            mjcf.set(attribute, mujoco_common.format_numbers([float(value)]))
        except (TypeError, ValueError):
            state["warnings"].append("The joint '%s' states %s as %r, which is not a number" % (name, key, value))
    reference = physics.get(SPRING_REFERENCE)
    if reference is None:
        return
    if ball:
        # MuJoCo's ball springs back to the orientation the body is written
        # in, and has nowhere to say another.
        state["warnings"].append(
            "The joint '%s' is written as a ball, which springs back to the orientation it is written in: its %s "
            "is not written" % (name, SPRING_REFERENCE)
        )
        return
    try:
        mjcf.set("springref", mujoco_common.format_numbers([joint_value(kind, float(reference))]))
    except (TypeError, ValueError):
        state["warnings"].append(
            "The joint '%s' states %s as %r, which is not a number" % (name, SPRING_REFERENCE, reference)
        )


def write_keyframe(mujoco, worldbody, state):
    """Write the model's start as ``<keyframe><key name="partcad:start">``, where a joint needs one.

    Only a bounded ball that starts away from its zero needs one (see
    'write_ball'): everything else is written where it starts, at MuJoCo's own
    initial configuration 'qpos0'. A key states the whole of 'qpos', so it is
    'qpos0' for every joint but those balls -- MuJoCo's numbering of it, read off
    the file as MuJoCo compiles it: bodies in the order they are written,
    depth first, and each body's joints in the order they are written in it.
    A free joint's coordinates are its body's 'pos' and 'quat', a hinge's and a
    slide's its 'ref' -- in radians and metres, since a key is not converted by
    '<compiler angle>' as a 'ref' is -- and a ball's the identity, or for a keyed
    one its start. Nothing is written for a model that needs no key.
    """
    if not state["keyed"]:
        return
    qpos = []

    def numbers(element, attribute, default):
        text = element.get(attribute)
        return [float(v) for v in text.split()] if text else list(default)

    def walk(body):
        for child in body:
            if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
                qpos.extend(numbers(body, "pos", (0.0, 0.0, 0.0)))
                qpos.extend(urdf_common.normalize(numbers(body, "quat", urdf_common.IDENTITY_Q)))
            elif child.tag == "joint" and child.get("type") == "ball":
                qpos.extend(state["keyed"].get(child.get("name"), urdf_common.IDENTITY_Q))
            elif child.tag == "joint":
                (ref,) = numbers(child, "ref", (0.0,))
                qpos.append(math.radians(ref) if child.get("type", "hinge") == "hinge" else ref)
        for child in body:
            if child.tag == "body":
                walk(child)

    walk(worldbody)
    keyframe = ElementTree.SubElement(mujoco, "keyframe")
    key = ElementTree.SubElement(keyframe, "key")
    key.set("name", START_KEY)
    key.set("qpos", mujoco_common.format_numbers(qpos, 17))


def write_equalities(mujoco, state):
    """Write what one freedom's value follows as an ``<equality><joint>``.

    Two kinds of it: within a joint, a screw's move follows its turn by its
    thread ('kinematics' couplings); and between joints, a 'mimic'. Both say
    'follower = ratio * leader + offset' in PartCAD's units. MJCF's joint
    equality is a polynomial about the two joints' 'ref's, in MuJoCo's own units
    (radians and metres, whatever the ``<compiler>`` says):

        y - y0 = a0 + a1 * (x - x0)

    so 'a1' is the ratio in those units and 'a0' is how far the starting values
    are from satisfying it -- nought for an assembly whose starting values agree
    with its own coupling.

    MuJoCo holds an equality the way it holds a contact: softly, by a stiff
    spring of its own (see 'solref'). Under a load a coupling gives a little --
    and how little depends on which way round it is written. MuJoCo softens a
    constraint by a share of what it takes to be the inverse inertia of the two
    joints, which it adds up without the coefficient between them; so a
    coupling written as "the slide follows the hinge by 1.6 mm a radian"
    (a1 = 0.0016) is softened by the light hinge's whole inverse inertia and
    barely holds at all -- a nut on a 10 mm thread fell 800 mm and turned a
    quarter turn -- while the same coupling written as "the hinge follows the
    slide by 628 radians a metre" holds to micrometres. A relation is the same
    relation either way round, so it is written with the coefficient of at
    least one: 'y - y0 = a0 + a1 (x - x0)' read backwards is
    'x - x0 = -a0 / a1 + (y - y0) / a1'.
    """
    for follower_name, mimic, followers in state["mimics"]:
        leader_name = str(mimic.get("joint") or "")
        leaders = state["joints"].get(leader_name) or []
        if len(followers) != 1 or len(leaders) != 1 or len(leaders[0]) != 1:
            state["warnings"].append(
                "The joint '%s' mimics '%s', and MJCF can say that only of one joint of one freedom following "
                "another: it is not written" % (follower_name, leader_name)
            )
            continue
        try:
            ratio = float(mimic.get("multiplier", 1.0))
            offset = float(mimic.get("offset", 0.0))
        except (TypeError, ValueError):
            state["warnings"].append("The joint '%s' states a mimic that is not numbers: %r" % (follower_name, mimic))
            continue
        (follower, freedom), (leader, leader_freedom) = followers[0], leaders[0][0]
        state["equalities"].append((follower, freedom, leader, leader_freedom, ratio, offset))

    if not state["equalities"]:
        return
    equality = ElementTree.SubElement(mujoco, "equality")
    for follower, freedom, leader, leader_freedom, ratio, offset in state["equalities"]:
        scale_y, scale_x = si_scale(freedom.kind), si_scale(leader_freedom.kind)
        a0 = (ratio * leader_freedom.value + offset - freedom.value) * scale_y
        a1 = ratio * scale_y / scale_x
        if not a1:
            state["warnings"].append(
                "'%s' follows '%s' with a ratio of nought, which holds it where it starts: not written"
                % (follower, leader)
            )
            continue
        if abs(a1) < 1.0:
            follower, leader, a0, a1 = leader, follower, -a0 / a1, 1.0 / a1
        element = ElementTree.SubElement(equality, "joint")
        element.set("joint1", follower)
        element.set("joint2", leader)
        element.set("polycoef", mujoco_common.format_numbers([a0, a1, 0.0, 0.0, 0.0]))


def write_excludes(mujoco, state):
    """Keep the two bodies of every joint from colliding with each other, as ``<contact><exclude>``.

    A joint is where two parts meet by design -- a pin in its bore, a carriage
    on its rail -- and what holds them together is the joint, not a contact.
    MuJoCo collides each body with the convex hull of the other's mesh, and a
    hull fills the bore the pin is in, so left to collide the two would be
    pushed apart by a contact the real mechanism does not have. MuJoCo filters a
    body's contacts with its parent of its own accord, but not where the parent
    is welded to the world -- a pendulum's frame, a robot bolted down -- which is
    exactly where a mechanism is most often simulated; so every joint says it.
    """
    if not state["excludes"]:
        return
    contact = ElementTree.SubElement(mujoco, "contact")
    for parent, child in state["excludes"]:
        exclude = ElementTree.SubElement(contact, "exclude")
        exclude.set("body1", parent)
        exclude.set("body2", child)


def emit_bodies(worldbody, found, state):
    """Write every body 'kinematics' found, nested along its parents, with its joint."""
    elements = {}
    for body in found:
        if isinstance(body.parent, kinematics.Body):
            parent = elements.get(id(body.parent))
            if parent is None:
                # Its parent held nothing and went back out; so does this.
                continue
            # Where the joint puts it at its starting values: the
            # configuration the assembly shows (but see 'written_location').
            pose = written_location(body)
        else:
            # Nothing in the tree holds it -- a free root, an element welded to
            # the world, or the child of a floating joint -- so it is placed in
            # the world, where the assembly shows it.
            parent, pose = worldbody, body.world
        emitted = emit_body(parent, body, pose, state)
        if emitted is None:
            continue
        element, own, shapes, inertial = emitted
        elements[id(body)] = element
        if body.joint is not None and isinstance(body.parent, kinematics.Body):
            write_joint(element, body, state)
            state["excludes"].append((parent.get("name"), element.get("name")))
        elif body.is_floating:
            # Free whatever 'static' says, and against what it was connected
            # to, which is written before it: 'bodies()' lists every body after
            # the one its joint is to.
            write_floating(element, body, elements.get(id(body.joint_parent)), state)
        elif body.parent is None and not state["options"]["static"]:
            # A body nothing holds: free to fall and to be pushed, which is
            # what makes a simulation of it worth running.
            ElementTree.SubElement(element, "freejoint")
        # A body welded to the world -- a 'fixed' scene element, or any root
        # under 'static' -- gets no joint: in MJCF that is what welded means.
        emit_geoms(element, own, shapes, inertial, body, state)


def add_ground_plane(worldbody, state):
    """The floor the model stands on: a static plane, and MuJoCo's own default."""
    geom = ElementTree.SubElement(worldbody, "geom")
    geom.set("name", state["geom_names"].derive("ground_plane", "ground_plane"))
    geom.set("type", "plane")
    geom.set("size", "0 0 0.05")
    geom.set("pos", "0 0 0")
    geom.set("rgba", "0.8 0.8 0.8 1")
    geom.set("condim", "3")


def add_light(worldbody):
    """The light a model needs to be looked at. It affects nothing physical."""
    light = ElementTree.SubElement(worldbody, "light")
    light.set("name", "sun")
    light.set("directional", "true")
    light.set("pos", "0 0 10")
    light.set("dir", "-0.5 0.1 -0.9")
    light.set("diffuse", "0.8 0.8 0.8")
    light.set("specular", "0.2 0.2 0.2")


def gravity_of(request, world):
    """The gravity to write, in m/s^2: this export's own option, the scene's, or MuJoCo's.

    An explicit 'gravity' on this export -- one a package configured on the
    file type, or one passed on the command line -- is a decision about this
    file and wins. Otherwise the scene's, which is what the scene says about its
    world. Otherwise 'DEFAULT_GRAVITY'. This package's own declaration states
    none, deliberately: a default there would be "explicit" too and would
    override every scene that ever stated one.
    """
    gravity = request.get("gravity") or world.get("gravity") or DEFAULT_GRAVITY
    return [float(v) for v in gravity]


def write_contact(option, request):
    """State how MuJoCo resolves friction, from the export's options or 'CONTACT_DEFAULTS'.

    An option set to null in a package's export configuration -- or passed as
    None -- leaves MuJoCo's own value, which is how a model that relies on the
    pyramidal cones gets them back.
    """
    for name, default in CONTACT_DEFAULTS.items():
        value = request.get(name, default)
        if value is None:
            continue
        option.set(name, value if isinstance(value, str) else mujoco_common.format_numbers([value], 6))


def write_medium(option, medium):
    """State the scene's fluid on ``<option>``, and return its density in kg/m^3 (or None).

    'medium' is what PartCAD resolved the scene's material to: 'density' in
    kg/m^3 and 'viscosity' in Pa*s, each present only when the material states
    it -- the units MJCF states them in, so they are written as they arrive. A
    positive value of either turns MuJoCo's passive fluid forces on (see the note
    above 'FLUID_INTEGRATOR'), and an empty one leaves ``<option>`` exactly as a
    scene in a vacuum has always had it.
    """
    density = medium.get("density")
    viscosity = medium.get("viscosity")
    fluid_density = float(density) if density else None
    if fluid_density:
        option.set("density", mujoco_common.format_numbers([fluid_density], 6))
    if viscosity:
        option.set("viscosity", mujoco_common.format_numbers([float(viscosity)], 6))
    if fluid_density or viscosity:
        option.set("integrator", FLUID_INTEGRATOR)
    return fluid_density


def process(path, request):
    root = tree_of(request)
    if not isinstance(root, dict) or not (
        ocp_serialize.is_shape_object(root) or ocp_serialize.is_assembly_object(root)
    ):
        raise ValueError("The MJCF exporter needs a shape or an assembly to export")

    model_dir = os.path.dirname(os.path.abspath(path)) or "."
    stem = os.path.splitext(os.path.basename(path))[0] or "model"
    model_name = model_name_of(request.get("model_name") or root.get("label") or stem, stem)
    mesh_dir_name = request.get("mesh_dir") or "%s_meshes" % stem
    mesh_dir = os.path.join(model_dir, mesh_dir_name)
    os.makedirs(mesh_dir, exist_ok=True)

    warnings = []
    if request.get("flatten") is not None:
        warnings.append(
            "'flatten' is no longer an option of this export, and is ignored: which nodes are bodies is PartCAD's "
            "body rule - every element of a scene a body of its own, everything in an assembly attached"
        )
    if request.get("ascii", False):
        warnings.append("MuJoCo reads binary STL only; 'ascii: true' produces meshes it will refuse to load")

    mujoco = ElementTree.Element("mujoco")
    mujoco.set("model", model_name)
    # Degrees, which is MJCF's own default and PartCAD's unit for a turn: a
    # hinge's 'ref', 'range' and 'springref' are written as the joint declares
    # them, and the file means the same whether or not whoever reads it honours
    # the element. Said all the same, so that nobody has to know the default.
    # Every orientation is a quaternion, which the setting does not touch.
    compiler = ElementTree.SubElement(mujoco, "compiler")
    compiler.set("angle", "degree")
    world = request.get("world") or {}
    option = ElementTree.SubElement(mujoco, "option")
    option.set("gravity", mujoco_common.format_numbers(gravity_of(request, world), 6))
    if request.get("timestep"):
        option.set("timestep", mujoco_common.format_numbers([request["timestep"]], 6))
    write_contact(option, request)
    fluid_density = write_medium(option, world.get("medium") or {})
    asset = ElementTree.SubElement(mujoco, "asset")
    worldbody = ElementTree.SubElement(mujoco, "worldbody")

    state = {
        "asset": asset,
        # One namespace per element type, as MuJoCo has them. A body or a joint
        # keeps PartCAD's name unless MJCF cannot have it, and is 'cube-2' after
        # 'cube' where it cannot, as PartCAD itself tells two apart; a geom is
        # 'cube_geom_1' and a mesh 'cube_1' after the first, as they always
        # were. A mesh's name is also its file's, so it answers to the
        # filesystem's rules rather than MJCF's.
        "body_names": naming.Transcoder(MJCF_BODY_NAMES),
        "joint_names": naming.Transcoder(MJCF_NAMES),
        "geom_names": naming.Transcoder(MJCF_NAMES, "_", 1),
        "mesh_names": naming.Transcoder(naming.FILE_NAMES, "_", 1),
        # The reported name of every MJCF joint -> where it is and whose
        # freedom it is, for the sidecar; see 'write_names'.
        "reported_joints": {},
        # PartCAD body -> the parts welded into it, for the sidecar; see
        # 'write_names'.
        "body_parts": {},
        "meshes": {},
        "mesh_dir": mesh_dir,
        "mesh_dir_name": mesh_dir_name,
        # Shape full name -> the properties its part declares ('physics',
        # 'material', 'color'). A part that came from a URDF, a world or an
        # MJCF model states its mass and friction here, and they go back out
        # rather than being recomputed.
        "properties": request.get("properties") or {},
        "unsupported": set(),
        # The scene's fluid's density, in kg/m^3, or None in a vacuum, and the
        # (body, centre of buoyancy) of each body buoyed in it. See
        # 'write_buoyancy'.
        "fluid_density": fluid_density,
        "centres_of_buoyancy": [],
        # PartCAD joint name -> the scalar MJCF joints written for each joint of
        # that name, as (MJCF name, freedom) pairs; what a 'mimic' looks its
        # leader up in. And what 'write_equalities' writes: couplings, and the
        # mimics to resolve once every joint is written.
        "joints": {},
        "equalities": [],
        "mimics": [],
        # (parent, child) MJCF body names of every joint; see 'write_excludes'.
        "excludes": [],
        # MJCF ball name -> its start in MuJoCo's terms, for every bounded ball
        # written at its zero rather than its start; see 'write_keyframe'.
        "keyed": {},
        "options": {
            "tolerance": request.get("tolerance", 0.1),
            "angularTolerance": request.get("angularTolerance", 0.1),
            "ascii": request.get("ascii", False),
            "static": request.get("static", True),
        },
        "warnings": warnings,
    }

    if request.get("light", True):
        add_light(worldbody)
    if request.get("ground_plane", True):
        add_ground_plane(worldbody, state)

    # The bodies, parents first, as PartCAD's body rule finds them in the tree;
    # what the rule set aside (a 'fixed' inside an assembly, a joint inside a
    # scene) is said rather than written.
    found = kinematics.bodies(root, request.get("properties"))
    warnings.extend(found.problems)
    # Every body PartCAD names is claimed before anything is written, so that
    # which of two bodies keeps a name MJCF can hold once does not depend on
    # the order they are nested in.
    state["body_names"].claim_all((body.name for body in found), "body")
    emit_bodies(worldbody, found, state)

    write_excludes(mujoco, state)
    write_equalities(mujoco, state)
    write_centres_of_buoyancy(mujoco, state)
    write_keyframe(mujoco, worldbody, state)

    if not state["meshes"]:
        warnings.append("Nothing was exported: the object holds no geometry MJCF can reference")

    renamed = naming.renamed_warning(MJCF_NAMES, state["body_names"], state["joint_names"])
    if renamed:
        warnings.append(renamed)

    ElementTree.indent(mujoco, space="  ")
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(ElementTree.tostring(mujoco, encoding="unicode"))
        f.write("\n")
    meshes = sorted("%s/%s.stl" % (mesh_dir_name, name) for name in set(state["meshes"].values()))
    # What was written besides 'path', relative to its directory: what lets
    # PartCAD cache this export and put it back whole (its 'files' result key).
    # Every name begins the way 'path' does - 'robot.xml' writes
    # 'robot_meshes/...' and 'robot.xml.names.json' - which is what lets a cached
    # model be put back under another name.
    files = list(meshes)
    if request.get("names_file"):
        write_names(path, model_name, root, state)
        files.append(os.path.basename(naming.sidecar_path(path)))
    else:
        naming.remove_sidecar(path)

    return {
        "success": True,
        "exception": None,
        "model_name": model_name,
        "mesh_dir": mesh_dir,
        "meshes": meshes,
        "files": files,
        "warnings": warnings,
        # Properties PartCAD holds and MJCF cannot state. Reported at info level
        # by the caller: nothing is wrong with the file, it just says less than
        # the package does.
        "unsupported": sorted(state["unsupported"]),
    }


def write_names(path, model_name, root, state):
    """Write the sidecar a simulation reads PartCAD's names back out of (see PartCAD's 'naming').

    Which MJCF name each PartCAD body and joint became, where each body's pose
    is read (its '<body>', by name) and each reported joint (its '<joint>', or
    a floating joint's '<freejoint>'), which PartCAD joint that is a freedom
    of, and its joint map ('"map"', 'kinematics.joint_map()'): what the MJCF
    joint is in PartCAD's terms, which a run turns MuJoCo's values back into
    PartCAD's by ('kinematics.reading()'), and which parts each body is welded
    from and where ('kinematics.parts_entry()'), which is how a run says which
    body a part is in. One file from the exporter to the
    simulation, as section 8.13.5 of PartCAD's simulation design record
    decided, rather than a map written into the model in a spelling of its own.
    The model's name is the scene's, which MJCF never needs to change but a
    reader should not have to know that.
    """
    bodies = {
        name: {"path": [internal], "parts": state["body_parts"].get(name, [])}
        for name, internal in state["body_names"].mapping().get("body", {}).items()
    }
    names = naming.merged_mapping(state["body_names"], state["joint_names"])
    names["scene"] = {str(root.get("label") or model_name): model_name}
    return naming.write_sidecar(path, "mjcf", names, bodies, dict(state["reported_joints"]))
