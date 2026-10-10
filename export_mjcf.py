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

Like PartCAD's URDF exporter, this one is handed the assembly *tree* itself
rather than the geometry it decodes to -- which is what ``decode: false`` on
this format's declaration asks for, see 'wrappers/wrapper_export.py' in PartCAD
itself. Decoding would keep the tree's shape and nothing
else about it: every node's ``name`` and ``label`` is dropped and its placement
is baked into the geometry rather than staying readable as data, and the bodies,
their poses and their properties are built from all three.

The mapping is the reverse of the MJCF reader's ('import_mjcf.py' beside this):

  * the exported object is the ``<mujoco>`` model,
  * anything placed in it is a ``<body>``, nested exactly as the tree nests,
  * a node that is geometry contributes a ``<geom type="mesh">`` referencing a
    mesh written next to the model file.

Three things this exporter decides that the file does not say, because a static
arrangement does not say them and a *simulation* needs all three:

  * **What moves.** MuJoCo bodies are welded to the world unless they carry a
    joint, so ``static: true`` (the default, and what a scene means) writes no
    joints at all. ``static: false`` gives every movable unit a ``<freejoint>``,
    which is what makes a simulation of one worth running.
  * **What a movable unit is.** With ``flatten: true`` every node that holds
    geometry becomes a body of its own directly in the ``<worldbody>``, at the
    world pose the tree puts it at. Without it the bodies nest as the tree does
    -- and a nested body with no joint between it and its parent is one rigid
    body with it, which is right for a rigid product and wrong for a stack of
    blocks that is meant to be able to fall over.
  * **What it stands on.** ``ground_plane`` and ``light`` are not part of what
    the scene says; they are what makes the model usable, the same way the world
    exporter's ``sun`` and ``ground_plane`` are. The plane is a ``<geom>`` of
    the ``<worldbody>`` itself, so it is static whatever ``static`` says.

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
import re
import sys
from xml.etree import ElementTree

# Pinned before anything that may pull OCP - see the note in ocp_serialize.
import pyexpat  # noqa: F401

# 'mujoco_common' is this package's, and sits beside this file. The other three
# are PartCAD's own and are the sandbox contract every implementation is written
# against: 'ocp_serialize' is the shape and assembly envelope format this is
# handed, 'urdf_common' is the pose arithmetic they are stated in, and
# 'mass_properties' is how several shapes' inertias add up into one body's. A
# sandbox runs a wrapper out of PartCAD's 'wrappers/' directory, so all three
# are already on sys.path by the time this is imported.
sys.path.append(os.path.dirname(__file__))
import mass_properties  # noqa: E402
import mujoco_common  # noqa: E402
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

# MJCF names end up as XML attributes and are referenced by name from geoms and
# from the simulation's own output, so anything outside this set is replaced.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

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


def sanitize_name(name, fallback):
    """An MJCF-safe name derived from a PartCAD name.

    PartCAD names carry package paths and ':' separators ("//pub/examples:logo")
    and the '/' that groups a link's own shapes, none of which belong in a name
    that a geom, a sensor or a keyframe refers to.

    A parameterized object's name also carries its parameter values, after a
    ';'. Those are dropped rather than spelled out: `pc sim` runs a scene whose
    subject *is* a parameter, so keeping them would make the model's name a
    transcription of the whole declaration -- and a name is read, not parsed.
    Which instance it is is what the run is about and is reported beside the
    file, not smuggled into it.
    """
    name = str(name or "").partition(";")[0]
    name = _UNSAFE_NAME.sub("_", name).strip("_")
    return name or fallback


class NameAllocator:
    """Hands out unique names, since MJCF requires them within an element type.

    Unique across the whole document rather than per type: that satisfies the
    requirement everywhere at once, and it keeps the names readable in what a
    simulation reports, which is where they are read from.
    """

    def __init__(self):
        self.used = set()

    def take(self, name, fallback="body"):
        base = sanitize_name(name, fallback)
        candidate = base
        suffix = 1
        while candidate in self.used:
            candidate = "%s_%d" % (base, suffix)
            suffix += 1
        self.used.add(candidate)
        return candidate


def node_geometry(node):
    """The node's own shape as a live TopoDS_Shape, in its own frame, or None.

    The placement carried by the node is deliberately left off: it becomes the
    pose of the element above the geometry, not part of the geometry.
    """
    if not ocp_serialize.is_shape_object(node):
        return None
    without_placement = {key: value for key, value in node.items() if key != ocp_serialize.KEY_LOCATION}
    return ocp_serialize.decode_shape(without_placement)


def write_mesh(shape, path, options):
    """Triangulate 'shape' and write it out as an STL file."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    BRepMesh_IncrementalMesh(
        shape,
        theLinDeflection=options["tolerance"],
        isRelative=True,
        theAngDeflection=options["angularTolerance"],
        isInParallel=True,
    )
    writer = StlAPI_Writer()
    writer.ASCIIMode = options["ascii"]
    if not writer.Write(shape, path) or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise Exception("Failed to write the mesh file: %s" % path)


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

    asset_name = state["mesh_names"].take(node.get("label") or body_name, "mesh")
    write_mesh(shape, os.path.join(state["mesh_dir"], "%s.stl" % asset_name), state["options"])
    mesh = ElementTree.SubElement(state["asset"], "mesh")
    mesh.set("name", asset_name)
    mesh.set("file", "%s/%s.stl" % (state["mesh_dir_name"], asset_name))
    mesh.set("scale", mujoco_common.format_numbers((MESH_SCALE,) * 3, 6))
    state["meshes"][key] = asset_name
    return asset_name


def shape_elements(node):
    """The (shape node, placement) pairs one body is built from, and its children.

    Usually a node is one shape and that is the whole of it. A sub-assembly
    whose children are named *under* it - "wrist" holding "wrist/1" and
    "wrist/2" - is one thing made of several shapes, and goes back out as one
    body with several ``<geom>`` elements rather than as a body per shape.

    The slash is the whole of the rule, and it is the only hierarchy PartCAD
    encodes in a name. It is the same rule the URDF and world exporters apply,
    and the same one all three readers write.
    """
    if ocp_serialize.is_shape_object(node):
        return [(node, None)], []

    children = node.get(ocp_serialize.KEY_ASSEMBLY, [])
    prefix = (node.get("label") or "") + "/"
    if not prefix.strip("/"):
        return [], children

    def belongs(child):
        return ocp_serialize.is_shape_object(child) and (child.get("label") or "").startswith(prefix)

    own = [(child, child.get(ocp_serialize.KEY_LOCATION)) for child in children if belongs(child)]
    return own, [child for child in children if not belongs(child)]


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


def physics_of(node, state):
    """What PartCAD says about the physics of the part behind one node of the tree."""
    return (state["properties"].get(node.get("name")) or {}).get("physics") or {}


def emit_geom(body, shape_node, placement, asset_name, physics, weighed, state, index):
    """Add one ``<geom>`` to a body, with what its part says about itself.

    'weighed' says whether the body has an ``<inertial>``. One that has not is
    one PartCAD could not weigh -- a mesh with no solid in it -- and MuJoCo
    weighs it instead, from the mesh, at the density PartCAD resolved for the
    part: its own, its material's, or the export's.
    """
    geom = ElementTree.SubElement(body, "geom")
    geom.set("name", state["geom_names"].take("%s_geom_%d" % (body.get("name"), index), "geom"))
    geom.set("type", "mesh")
    geom.set("mesh", asset_name)
    if placement is not None:
        pose = urdf_common.from_packed(placement)
        if not urdf_common.is_identity(pose):
            geom.set("pos", mujoco_common.format_pos(pose))
            geom.set("quat", mujoco_common.format_quat(pose))

    properties = state["properties"].get(shape_node.get("name")) or {}
    rgba = mujoco_common.color_rgba(properties.get("color"))
    if rgba is not None:
        geom.set("rgba", rgba)

    if not weighed:
        density = (properties.get("physics") or {}).get("density")
        if density is not None and float(density) > 0.0:
            geom.set("density", mujoco_common.format_numbers([float(density)], 6))

    for name, (attribute, render) in GEOM_PHYSICS.items():
        if name not in physics:
            continue
        value = render(physics[name])
        if value is not None:
            geom.set(attribute, value)
    return geom


def emit_body(parent, node, pose, elements, children_present, state):
    """Add one ``<body>``, with a geom per shape it holds. Returns the element."""
    body_name = state["body_names"].take(node.get("label") or node.get("name"), "body")
    body = ElementTree.SubElement(parent, "body")
    body.set("name", body_name)
    if not urdf_common.is_identity(pose):
        body.set("pos", mujoco_common.format_pos(pose))
        body.set("quat", mujoco_common.format_quat(pose))

    physics = physics_of(node, state)
    shapes = [
        (index, shape_node, placement, node_geometry(shape_node))
        for index, (shape_node, placement) in enumerate(elements)
    ]
    shapes = [entry for entry in shapes if entry[3] is not None]

    # One shape is the body, as PartCAD resolved it. Several are added up, each
    # where the body holds it -- unless the body states its own mass, which
    # beats the sum of its pieces. See 'mass_properties.of_body()' in PartCAD.
    own = None if len(elements) == 1 and elements[0][0] is node else physics
    parts = [(physics_of(shape_node, state), placement) for _, shape_node, placement, _ in shapes]
    inertial = carried_inertial(mass_properties.of_body(parts, own=own))
    if inertial is not None:
        write_inertial(body, inertial)

    if not state["options"]["static"] and parent.tag == "worldbody":
        # What makes a simulation of this model worth running: a body with no
        # joint is welded to the world and can neither fall nor be pushed.
        ElementTree.SubElement(body, "freejoint")

    written = 0
    for index, shape_node, placement, shape in shapes:
        asset_name = mesh_asset(shape, shape_node, body_name, state)
        emit_geom(body, shape_node, placement, asset_name, physics, inertial is not None, state, index)
        written += 1
    if written:
        write_buoyancy(body, inertial, parts, state)

    if not written and not children_present:
        # A frame with nothing in it and nothing under it. MuJoCo accepts an
        # empty body, but it is noise in the model and in what a simulation
        # reports, so it goes back out.
        parent.remove(body)
        state["body_names"].used.discard(body_name)
        return None

    state["unsupported"].update(set(physics) - MJCF_STATED)
    return body


def emit_nested(node, parent, pose, state):
    """Add 'node' and its subtree under 'parent', mirroring the tree's nesting."""
    elements, children = shape_elements(node)
    body = emit_body(parent, node, pose, elements, bool(children), state)
    if body is None:
        return
    for child in children:
        emit_nested(child, body, urdf_common.from_packed(child.get(ocp_serialize.KEY_LOCATION)), state)


def emit_flat(node, worldbody, pose, state):
    """Add every node that holds geometry as a body of the ``<worldbody>`` itself.

    Each one is placed at the world pose the tree puts it at, so the arrangement
    is unchanged -- what changes is that the bodies are now independent of each
    other, which is what lets them move independently. See the module docstring.
    """
    elements, children = shape_elements(node)
    if elements:
        emit_body(worldbody, node, pose, elements, False, state)
    for child in children:
        emit_flat(
            child,
            worldbody,
            urdf_common.compose(pose, urdf_common.from_packed(child.get(ocp_serialize.KEY_LOCATION))),
            state,
        )


def add_ground_plane(worldbody, state):
    """The floor the model stands on: a static plane, and MuJoCo's own default."""
    geom = ElementTree.SubElement(worldbody, "geom")
    geom.set("name", state["geom_names"].take("ground_plane", "ground_plane"))
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
    root = request["wrapped"]
    if not isinstance(root, dict) or not (
        ocp_serialize.is_shape_object(root) or ocp_serialize.is_assembly_object(root)
    ):
        raise ValueError("The MJCF exporter needs a shape or an assembly to export")

    model_dir = os.path.dirname(os.path.abspath(path)) or "."
    stem = os.path.splitext(os.path.basename(path))[0] or "model"
    model_name = sanitize_name(request.get("model_name") or root.get("label") or stem, stem)
    mesh_dir_name = request.get("mesh_dir") or "%s_meshes" % stem
    mesh_dir = os.path.join(model_dir, mesh_dir_name)
    os.makedirs(mesh_dir, exist_ok=True)

    warnings = []
    if request.get("ascii", False):
        warnings.append("MuJoCo reads binary STL only; 'ascii: true' produces meshes it will refuse to load")

    mujoco = ElementTree.Element("mujoco")
    mujoco.set("model", model_name)
    # Radians, so that what is written needs no <compiler> to be read back the
    # way it was meant -- MJCF's own default is degrees, and every angle here
    # is a quaternion anyway.
    compiler = ElementTree.SubElement(mujoco, "compiler")
    compiler.set("angle", "radian")
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
        "body_names": NameAllocator(),
        "geom_names": NameAllocator(),
        "mesh_names": NameAllocator(),
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

    emit = emit_flat if request.get("flatten", False) else emit_nested
    root_pose = urdf_common.from_packed(root.get(ocp_serialize.KEY_LOCATION))
    if ocp_serialize.is_shape_object(root):
        emit(root, worldbody, root_pose, state)
    else:
        # The exported object *is* the model, so its children are its top-level
        # bodies, each carrying the exported object's own placement on top of
        # its own.
        for child in root.get(ocp_serialize.KEY_ASSEMBLY) or []:
            child_pose = urdf_common.compose(root_pose, urdf_common.from_packed(child.get(ocp_serialize.KEY_LOCATION)))
            emit(child, worldbody, child_pose, state)

    write_centres_of_buoyancy(mujoco, state)

    if not state["meshes"]:
        warnings.append("Nothing was exported: the object holds no geometry MJCF can reference")

    ElementTree.indent(mujoco, space="  ")
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(ElementTree.tostring(mujoco, encoding="unicode"))
        f.write("\n")

    return {
        "success": True,
        "exception": None,
        "model_name": model_name,
        "mesh_dir": mesh_dir,
        "meshes": sorted("%s/%s.stl" % (mesh_dir_name, name) for name in set(state["meshes"].values())),
        "warnings": warnings,
        # Properties PartCAD holds and MJCF cannot state. Reported at info level
        # by the caller: nothing is wrong with the file, it just says less than
        # the package does.
        "unsupported": sorted(state["unsupported"]),
    }
