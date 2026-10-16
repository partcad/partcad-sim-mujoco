# partcad-sim-mujoco

A [PartCAD](https://partcad.org/) package that is **everything PartCAD knows
about [MuJoCo](https://mujoco.org/)**: reading a model, writing one, opening one,
and running one — so that a part or an assembly can state what it is supposed to
do once the world is switched on, and have that checked.

```yaml
dependencies:
  sim-mujoco:
    type: git
    url: https://github.com/partcad/partcad-sim-mujoco.git

scenes:
  table:
    # A model somebody else wrote, used directly as a PartCAD scene.
    type: sim-mujoco:mjcf
    path: table.xml

assemblies:
  stack:
    type: assy
    simulate:
      stands:
        simulation: sim-mujoco:mujoco
        offset: [[0, 0, 10], [0, 0, 1], 0]
        validation: |
          max(
              abs(after["bodies"][name]["pos"][2] - before["bodies"][name]["pos"][2])
              for name in before["bodies"]
          ) < 2.0
```

```console
$ pc sim -a stack
INFO: //your/package:stack: the simulation 'stands' validated
```

> [!IMPORTANT]
> Naming this package's reader by its full path — `type: sim-mujoco:mjcf` — needs
> a PartCAD carrying [partcad/partcad#643](https://github.com/partcad/partcad/pull/643).
> Earlier releases look the package up from the root while the package declaring
> the object is still loading, find nothing, and record the object as broken. The
> `simulation:` entry works on any release.

## Four entry points, one format

MuJoCo describes a model in **MJCF**, an XML dialect of its own. This package
declares all four of the things a PartCAD package can teach PartCAD, and all
four are about that one format:

| Section | What it does | How it is asked for |
| --- | --- | --- |
| `import:` | read an `.xml` MJCF model as a PartCAD object | `type: sim-mujoco:mjcf` |
| `export:` | write a PartCAD scene or assembly out as one | `pc export -t sim-mujoco:mjcf` |
| `simulation:` | run one and say where everything ended up | `simulation: sim-mujoco:mujoco` |
| `open:` | open one in the MuJoCo viewer | `pc ide open --with mujoco` |

They are declared together because they are one piece of knowledge. A reader and
a writer of the same format disagree the moment they are maintained apart, and
the simulator is what decides what the file has to say in the first place.

MJCF is the one arrangement format routinely used for **both** an assembly and a
scene, and nothing in the file says which: an MJCF model of a robot arm is an
assembly, an MJCF model of the table it stands on is a scene. So the reader
declares both, and the section a package puts it in is what it meant.

`pc convert scene` moves a package between MJCF and PartCAD's own `assy`:

```shell
pc convert scene -t assy table               # take it over, as parts of your own
pc convert scene -t sim-mujoco:mjcf bench    # and write it back out
```

## What a run reports

PartCAD exports the scene as MJCF -- every body
nothing holds free to move, every joint written (see [Mechanisms](#mechanisms))
and a ground plane under it -- hands this package the file, and this package
steps the model for `duration` seconds of simulated time. It then
reports where every body was at the start and at the end:

```json
{
  "before": {"time": 0.0,  "bodies": {"top": {"pos": [0, 0, 30], "quat": [1, 0, 0, 0]}}, "joints": {}},
  "after":  {"time": 10.0, "bodies": {"top": {"pos": [29.7, 0, 9.8], "quat": [...]}}, "joints": {}},
  "simulator": "mujoco", "duration": 10.0, "steps": 5000, "units": "mm"
}
```

Positions are in **millimetres**, PartCAD's unit everywhere, not the metres
MuJoCo works in: a `validation:` expression is written by whoever wrote the
part, against the numbers that part is drawn in.

### Joints

A model with joints in it — a hinge, a slide, a ball — also says where each of
them is, as `joints` beside `bodies`, keyed by the name the MJCF gives the
joint and stated the way an interface's `motion:` is: **degrees** for a turn,
**millimetres** for a move.

```json
"joints": {
  "swing": {"type": "continuous", "pos": 90.0, "vel": 358.9, "effort": 0.0},
  "lift":  {"type": "prismatic", "pos": -859.6, "vel": -4106.7, "effort": 0.0},
  "wrist": {"type": "ball", "quat": [0.707, 0, 0.707, 0], "vel": [0, 358.9, 0], "effort": [0, 0, 0]}
}
```

| `type` | in MJCF | `pos` | `vel` | `effort` |
| --- | --- | --- | --- | --- |
| `revolute` | a hinge with a `range` | degrees | deg/s | N·m |
| `continuous` | a hinge without one | degrees | deg/s | N·m |
| `prismatic` | a slide | mm | mm/s | N |
| `ball` | a ball | `quat` (`w x y z`) instead | `[x, y, z]` deg/s | `[x, y, z]` N·m |

* **`pos`** is the joint's own coordinate: zero where the model placed the body,
  or at the joint's `ref` where it states one. A pendulum written out
  horizontal reads 0 there and 90, one way or the other, hanging down. A model
  this package's exporter wrote states every hinge's and slide's starting value
  as its `ref`, so there `pos` is the joint's value as PartCAD states it.
* A ball joint's **`quat`** is its turn away from where the model placed the
  body, and its `vel` is the angular velocity about the axes of the body's own
  frame — MuJoCo's own convention for it.
* **`effort`** is what the model's actuators exert along the joint — MuJoCo's
  `qfrc_actuator`, gear and force limits applied — which is the quantity an
  interface's `maxEffort` bounds. It is zero for a joint nothing drives. It is
  deliberately not the constraint force: what a limit or a contact pushes back
  with is a reaction, and whether a motor was strong enough is a question about
  the action.
* A **free joint** is not listed. Its coordinate is the body's own position and
  orientation, which `bodies` states already, in the same units. So a model of
  free bodies — a scene of parts — reports `"joints": {}`, and a validation can
  walk `after["joints"]` without asking first whether it is there.
* A joint the MJCF does not name is `joint_<n>`, `n` being its index in the
  compiled model: the rule `bodies` follows.
* Joints are read at exactly `time`. Bodies are read where MuJoCo last
  evaluated them, which after a step is up to one step earlier (2 ms at
  MuJoCo's default) — so once anything moves, a joint's angle and its body's
  orientation can disagree by a step's worth of motion.

```yaml
validation: after["joints"]["swing"]["pos"] > 80    # it swung down
```

[partcad-sim-gazebo](https://github.com/partcad/partcad-sim-gazebo) reports
joints in the same vocabulary, so a validation of where a joint went reads the
same against either engine. Gazebo publishes neither an effort nor a ball
joint's orientation, so those two are MuJoCo's alone.

PartCAD reads nothing inside `before` and `after`. It hands them to the
`validation:` expression the package wrote and reports what that says — every
judgement in that sentence belongs to the package being simulated.

Nothing has to be installed by hand. PartCAD installs this package's Python
requirements into the sandbox it runs the implementation in, so a machine with
no MuJoCo on it simulates just the same.

## Mechanisms

Which parts of a scene are one rigid body, and which move against which, is not
this package's decision. It is PartCAD's **body rule**, in PartCAD's own
`kinematics` module, which the exporter calls and writes the answer of: every
element of a scene is a body of its own, everything inside an assembly is
attached to it — welded into one body, whose mass and inertia are its parts'
added up — and a connection that keeps a degree of freedom (`motion:` on an
interface, a mating or the connection) is a **joint**. An assembly placed in a
scene is one tree of bodies, nested in the MJCF as it hangs together, with its
root free; a scene element marked `fixed: true` is welded to the world.

Each joint is written as MuJoCo's own, by what `kinematics` classifies it as:

| The joint is | In MJCF |
| --- | --- |
| one turn, limited (`revolute`) | a `hinge` with a `range` |
| one turn, unlimited (`continuous`) | a `hinge` without one |
| one move (`prismatic`) | a `slide` |
| a turn and a move on one axis (`cylindrical`) | a `hinge` and a `slide` |
| the same, coupled by a thread (`screw`) | a `hinge`, a `slide` and an `<equality><joint polycoef>` |
| two turns through a point (`universal`) | two `hinge`s |
| three turns through a point (`ball`) | a `ball` at the point |
| two moves and the turn about their normal (`planar`) | two `slide`s and a `hinge` |
| all six (`floating`) | three `slide`s and a `ball`: a `<freejoint>` is allowed only directly in the `<worldbody>` |
| anything else (`chain`) | its freedoms in order, a `hinge` or a `slide` each, in the one body |

* **Names.** A joint written as one MJCF joint has exactly the name PartCAD
  gives it — `<child link>-<target link>`, or the connection's `joint:` — so a
  validation reads `after["joints"]["shoulder_pan"]`. One written as several
  names each after the parameter it is: `nut-bolt.turnZ`, `nut-bolt.moveZ`, and
  a floating joint's `….ball`. A body is named as PartCAD names it: after the
  link that placed it, `leg`, `leg-2`.
* **Where it starts.** Every body is written where the assembly shows it — at
  the joints' starting values, `toParams` — and every hinge's and slide's `ref`
  is that value, so MuJoCo's zero is PartCAD's zero and a reading of the joint
  is its value as PartCAD states it. `pos` and `axis` are in the body the joint
  moves, as MJCF states them. A ball has no `ref`: it starts at the
  orientation the body is written in, and reads its turn away from it.
* **Limits.** A freedom's bounds are its `range`, in degrees for a hinge
  (`<compiler angle="degree">`, which every model this package writes says) and
  metres for a slide; unlimited is `limited="false"` and no range. A bound on
  one side only is written with the other a million degrees (or a kilometre)
  away, since a range has two ends. An MJCF ball can bound only the whole angle
  it turns through, so a ball's three turns' limits are reported, not written.
* **Couplings.** A screw's move follows its turn by its `threadStep`, and a
  `mimic` follows another joint; both are an `<equality><joint>` whose
  `polycoef` is the ratio in MuJoCo's own units (radians and metres) about the
  two `ref`s. It is written with the coefficient of at least one — "the hinge
  follows the slide by 628 radians a metre" rather than "the slide follows the
  hinge by 1.6 mm a radian" — because MuJoCo softens an equality by the two
  joints' inverse inertia, without the coefficient: written the other way, a nut
  on a 10 mm thread fell 800 mm and turned a quarter turn. A `mimic` between
  joints of more than one freedom is reported, not written.
* **Contacts.** The two bodies of a joint are kept from colliding with each
  other (`<contact><exclude>`): a pin sits in its bore by design, and MuJoCo
  collides each body with the *convex hull* of the other's mesh, which fills the
  bore. MuJoCo filters a child's contacts with its parent of its own accord, but
  not where the parent is welded to the world — a pendulum's frame, a robot
  bolted down.
* **What moving it costs.** A joint's `physics:` is written per unit of its own
  motion in SI, which is what MuJoCo takes: `damping` (N·m·s/rad, N·s/m) as
  `damping`, `friction` (N·m, N) as `frictionloss`, `springStiffness` as
  `stiffness` and `springReference` as `springref`. A screw's coupled move gets
  none, since its turn already pays for it. **Not written**, and listed as
  unsupported: `maxEffort` — MJCF's `actuatorfrcrange` clamps what *actuators*
  apply, and a model PartCAD writes has none — `maxVelocity` (MuJoCo limits no
  joint's velocity), `softLimits`, and the ODE solver settings URDF's
  `<gazebo>` carries. Nothing is written for MJCF's `armature` either: it is a
  motor's rotor inertia, and PartCAD's joints state none.

Two export options decide the rest:

| Option | Default | What it does |
| --- | --- | --- |
| `static` | `true` | Weld every body nothing holds to the world: a file written to be looked at keeps the scene where it was drawn. Joints inside an assembly are written either way, so a pendulum opened in MuJoCo's viewer still swings. The simulation asks for `static: false`, which gives each such body a `<freejoint>`. |
| `flatten` | — | Gone. It made every node holding geometry a free body of its own, which the body rule now says of every element of a scene and of nothing inside an assembly. Passing it is reported. |

## Snapshots

PartCAD asks every run for two pictures of the scene — one before anything
moved, one when the time is up — and says where to take them from: the scene's
own `render: png:` viewpoint (`viewport_origin`, `viewport_up`), the same one
`pc render` uses, or the corner a rendered part is drawn from when nothing says
otherwise. This package draws them, writes them into the run directory and
reports them beside the result:

```json
{"snapshots": {"before": "snapshot-before.png", "after": "snapshot-after.png"}}
```

They are what the PartCAD IDE's **Validation → Simulation** tab shows side by
side, and `pc sim` says where they were written.

They are MuJoCo's own compiled geometry — its meshes and its primitives — at
MuJoCo's own poses, rasterized on the CPU with numpy (`snapshot_raster.py`).
Not MuJoCo's OpenGL renderer: a headless OpenGL needs EGL or OSMesa, native
libraries no wheel carries and PartCAD's sandbox images do not have, and a
picture that only a machine with a GPU driver set up can take is one most runs
would never get. Both pictures are framed alike, on everything either of them
holds, over a floor whose squares are fixed to the world — so a block that fell
is lower in the frame and further along the floor, rather than re-centred.

A picture that cannot be drawn is a warning, never a failed run: the verdict
does not depend on it.

## Parameters

Set any of these as fields of the simulation (per package), or in the `params:`
of one `simulate:` entry (per simulation).

| Parameter | Default | What it is |
| --- | --- | --- |
| `duration` | `10.0` | Seconds of simulated time to run for. |
| `timestep` | MuJoCo's | Integration step, in seconds. |
| `gravity` | the scene's | m/s², in the scene's frame. Overrides the scene's `gravity:` for this run; unset, the run is under the scene's, or MuJoCo's `[0, 0, -9.81]` when the scene states none. |
| `samples` | `0` | Report the state at this many evenly spaced instants too, as `samples`. |
| `timeout` | PartCAD's | Wall-clock seconds the run may take. PartCAD works it out and stops the run when it is spent; see below. |

A run is refused before anything is stepped -- naming the parameter -- when it
asks for a `duration` or a `timestep` that is not a positive number, a step of
more than a second, more than 50 million steps (`duration` / `timestep`), or
more than 10,000 `samples`. Each of those is a typo turning into hours of
stepping or gigabytes of result, and a run quietly clamped to something else
would be validated against a run nobody asked for.

## When a run goes wrong

MuJoCo does not stop a model whose state stops being a number. It prints a
warning once, resets the state, and goes on -- and the clock may stop advancing
with it, so a run that steps until `duration` never ends, while the state it is
left in is a finite number that means nothing. A validation of "nothing fell
over" could pass on it. So every step is checked against MuJoCo's own
instability warnings (`mjWARN_BADQPOS`, `mjWARN_BADQVEL`, `mjWARN_BADQACC`),
and a run that diverges fails at once, saying when, and at which joint of which
body, in MuJoCo's words as well as this plugin's:

```text
MuJoCo's simulation diverged at t = 0.734 s: the acceleration of joint 'swing' (body 'arm') is not a finite
number (mjWARN_BADQACC). [...] A smaller 'timestep', an integrator that is more stable for stiff models
('implicitfast'), and a mass or an inertia that is not near zero are the usual remedies. MuJoCo said: Nan,
Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 0.7340.
```

MuJoCo's warnings are kept for the result rather than printed, so nothing is
written to a `MUJOCO_LOG.TXT` in the plugin's directory. The ones that do not
end a run -- out of room for contacts or constraints, an inertia that is not
physical, a control that is not a number -- are reported in `warnings`.

Two bounds are under that, for a run that will not end for any other reason:

- **Steps.** A run that has taken twice the steps its `duration` takes has been
  taken back in time, and is stopped.
- **Wall clock.** PartCAD hands every run its budget as `timeout` (see "When a
  run hangs or crashes" in PartCAD's simulation guide) and stops the run when it
  is spent. This script stops a tenth of it earlier -- at least a second, at
  most thirty -- and says how far it got: `MuJoCo reached 41.2 s of the 600 s
  asked for in 270.3 s of wall-clock time, and the run's budget is 300 s`.
  Give a slow run longer with `params: {timeout: ...}`, or shorten its
  `duration`.

## Friction is a fact about the material

Whether a stack of blocks stands up is not a property of its geometry. Two 20 mm
cubes, one squarely on the other, in a world whose gravity is
tilted 15° off vertical -- a ramp with no edge to slide off, `tan 15° = 0.268`
-- for ten seconds, nothing changed but the sliding coefficient both blocks are
given (measured with this package's exporter and simulation):

| sliding friction | what happens to the top block |
| --- | --- |
| 0.04 (PTFE) | slides 69 mm and falls 20 mm, onto the floor |
| 0.1 | slides 63 mm and falls |
| 0.2 | slides 56 mm and falls |
| 0.25 | slides 32 mm and falls |
| 0.28 | slides 34 mm and falls |
| 0.3 | stays: 0.5 mm of creep, 0.4 mm of settling |
| 0.4 | stays: 0.5 mm |
| 1.05 (dry aluminium) | stays: 0.5 mm |

On a level floor every one of them stays put, PTFE included: nothing pushes a
block sideways, so its friction is never asked anything. The threshold MuJoCo
finds, between 0.28 and 0.3, is a little above `tan 15°` because its contacts
are soft; see below.

So state it. A part that declares a material whose `mu` is set gets that
coefficient written into the MJCF, and the simulation answers for the material
the part is actually made of. A part that states neither gets MuJoCo's default
of 1.0 — a plausible number for metal on metal, a badly wrong one for PTFE, and
in either case a number nobody chose.

### How MuJoCo combines the two sides of a contact

MuJoCo takes the **element-wise maximum** of the two geoms' friction
coefficients, unless one geom has a higher `priority`, in which case its
coefficients are used ([Contact
parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)).
This exporter sets no priority, so:

* **block on block** -- two parts of one material meet at that material's
  `mu`; two of different materials at the larger of the two;
* **block on floor** -- at no less than 1.0, because the ground plane states no
  friction and so has MuJoCo's default. A PTFE block grips the floor; it is a
  PTFE block on another PTFE block that slides.

### Contacts that stick

MuJoCo's contacts are soft, and a block under a steady sideways load slips at a
small steady rate even when the load is well inside its friction cone. With
MuJoCo's own defaults -- pyramidal cones, `impratio` 1, no NoSlip pass -- that
rate is large: the same aluminium stack, which should not move at all, slid
33 mm at a 6° tilt and lost its top block, and at 15° the bottom block slid
18 mm along the floor too. So every model this package writes carries MuJoCo's
own remedies ("Preventing slip" in its modelling guide):

```xml
<option cone="elliptic" impratio="10" noslip_iterations="3" />
```

which hold the aluminium stack to the half-millimetre of creep in the table
above. They are export parameters (`cone`, `impratio`, `noslip_iterations` on
`mjcf`); set one to null for MuJoCo's own value.

## So is what it weighs

The same material says what the part weighs. PartCAD works out every part's
mass, centre of mass and inertia -- what the part states, or else its solid at
the density of what it is made of -- and hands them over with the part, so a
body's `<inertial>` is exactly what `pc info` reports for it, and what the URDF
and SDFormat exports of the same object say. A PTFE block weighs what PTFE
weighs rather than what aluminium does. The order is:

1. what the part states, as stated;
2. its `density` -- one it states, or else its material's;
3. the export's `density` parameter;
4. 2700 kg/m³, aluminium.

This exporter works none of that out. A body of several shapes -- two
materials in one link -- is added up by PartCAD's `mass_properties`, the one
copy of that arithmetic all three exporters share, so it balances where it
should. A body PartCAD could not weigh, because its mesh has no solid in it, is
left to MuJoCo: its geoms carry the density PartCAD resolved, and MuJoCo
integrates the mesh at it.

It needs the PartCAD that does this, and says so: `partcad:` in `partcad.yaml`
makes an older one refuse the package.

## Gravity, and the fluid a scene is filled with

A PartCAD scene may say what its world is like, beside where things are in it:

```yaml
scenes:
  tank:
    type: assy
    gravity: [0, 0, -9.81]                                # m/s², the scene's own frame
    medium: //pub/std/manufacturing/material/fluid:water  # a material, by reference
```

PartCAD resolves the material and hands this exporter its density and
viscosity. Every one of these is SI in PartCAD and in MJCF alike, so nothing is
converted on the way:

| The scene says | The MJCF says |
| --- | --- |
| `gravity` (m/s²) | `<option gravity>` (m/s²) |
| the medium's `density` (kg/m³) | `<option density>` (kg/m³) |
| the medium's `viscosity` (Pa·s) | `<option viscosity>` (Pa·s) |
| the medium's `density` again | each body's `gravcomp`: ρ_fluid · V / m |
| where the body displaces it | `<custom><numeric name="partcad:centre_of_buoyancy:<body>">`: the centre of buoyancy, m, in the body's frame |

`m` is the mass on the body's `<inertial>`, which PartCAD resolved, and `V` the
volume PartCAD measured the body's solids to enclose, handed over beside it and
added up by PartCAD's `mass_properties.volume_of()`; the displaced mass is
`mass_properties.mass_of(V, ρ_fluid)`. This exporter measures and weighs
nothing itself. A body PartCAD could not weigh, or whose volume it does not know
(an open mesh), gets no buoyancy, and a warning says which.

A model with a fluid in it is also written with `integrator="implicitfast"`,
which is what MuJoCo recommends wherever velocity-dependent forces act. A scene
that states neither gets exactly the `<option>` it always did — MuJoCo's own
gravity, in a vacuum.

**What MuJoCo models, and what it does not.** `<option density viscosity>` turns
on MuJoCo's [passive fluid model](https://mujoco.readthedocs.io/en/stable/computation/fluid.html)
in its default, *inertia-box* form: each body is taken to be the box with its
mass and inertia, and the fluid resists its motion with a drag quadratic in its
speed (from the density) and a Stokes resistance linear in it (from the
viscosity), with the matching torques. That model has **no buoyancy** at all —
a block of foam would sink, only slower — so this package adds it, through
MuJoCo's own `gravcomp`: an upward force at the body's centre of mass of that
fraction of its weight, which MuJoCo's documentation calls "a buoyancy effect"
when it is above one. The fraction is the fluid displaced over the body's own
mass, so it follows any `gravity` a run is given. What that leaves out:

* **A surface.** The fluid fills the whole world. A body lighter than it rises
  for as long as the run lasts, at the speed its drag allows, rather than coming
  to float at a waterline.
* **The centre of buoyancy, outside `pc sim`.** The lift acts at the centre of
  buoyancy -- the centroid of what the body displaces, PartCAD's
  `centerOfVolume` combined per body by `mass_properties.displacement_of()` --
  which is what rights a body whose weight is not centred where its volume is:
  a hull with a heavy keel, a float that states a low `centerOfMass`. MJCF has
  no way to say where a force acts, so `gravcomp` applies it at the centre of
  mass and the simulation adds the moment `(r_buoyancy − r_mass) × lift` to the
  body every step, through `xfrc_applied`, from the centre of buoyancy the
  exporter writes as a custom numeric. The same model opened in MuJoCo's viewer
  floats but does not right itself.
* **Partial submersion.** There is no surface, so nothing is ever half in the
  water. A waterline would need the volume below it and that volume's centroid
  every step, out of the mesh -- geometry the simulation would have to do
  itself, which nothing PartCAD hands over could answer in advance.
* **Sealed cavities.** The volume displaced is the solid's, so a hollow part is
  buoyed as if flooded. A float is drawn as the solid it displaces, and states
  its own `mass`; PartCAD scales the solid's inertia to it.
* **Added mass, lift and the Magnus effect.** Those are MuJoCo's per-geom
  *ellipsoid* model (`fluidshape="ellipsoid"`), which needs five coefficients
  per geom that nothing in a PartCAD scene states. It is not written.

**Which gravity wins.** In order: a `simulate:`'s own `params: {gravity: ...}`,
for that run; the scene's `gravity:`; MuJoCo's `[0, 0, -9.81]`. An explicit
`gravity` on the `mjcf` export does the same for a file written with
`pc export`. Neither this package's export nor its simulation states a default
of its own any more — until PartCAD scenes could say anything, a default there
was harmless, and now it would beat every scene that does.

## Tests

```shell
pytest
```

`test_mjcf.py` needs `partcad` installed: the reader and the writer are written
against the sandbox contract that lives there — `ocp_serialize`, `urdf_common`,
`mass_properties` and `primitive_shapes` — and the tests import them the same
way a sandbox does. CI installs it from PartCAD's `devel` branch, so that a
change to that contract fails here before it is released.
It needs no MuJoCo, because reading and writing a model do not involve one.

`test_snapshot.py` needs numpy for the pictures, and MuJoCo for the half that
runs the simulation end to end — a stack whose top block falls off, drawn
before and after. That half is skipped where MuJoCo is not installed; CI
installs it.

`test_export_joints.py` exports the trees PartCAD itself builds for every kind
of joint — a package of cubes composed by PartCAD's own ASSY code, with a
stand-in for each part's geometry — loads each in MuJoCo, and checks that every
body is where PartCAD's `kinematics` says, at the joints' `ref` and moved away
from it. It checks the physics against closed forms too: a 300 mm bar on a
revolute joint released horizontal passes the bottom at √(2·m·g·d / I) and,
swinging 5°, has the period 2π·√(I / (m·g·d)); a carriage on a slide falls
½·g·t²; a nut on a screw turns as its thread says.

`test_safety.py` proves each of the nets in "When a run goes wrong" with a
run that needs it: a stiff pendulum that diverges on its first step (and, run
without the check, a thousand steps that never get past 10 ms), a run of ten
thousand simulated seconds under a budget of a second and a half, a run that
MuJoCo keeps taking back to nought seconds, and every absurd parameter. It needs
MuJoCo for the runs.

`test_joints.py` runs a hand-written model — a pendulum on a hinge and on a
ball joint, a carriage on a slide, one an actuator holds up — and checks the
`joints` it reports against physics with a known answer: the speed a pendulum
released from horizontal passes the bottom at, the quarter period it takes to
get there, a fall along a slide in millimetres. It needs MuJoCo too.

## Where this came from

The simulation used to ship inside the `partcad` wheel as `//builtin/simulate`,
and the reader, the writer and the `open:` entry did until 0.8.80. None of them
should: PartCAD ships the *concept* — the `simulate:` section, the sandbox
wrapper, the runner, the sections a package declares — and a simulator is
somebody's program with a release cycle of its own. Pinning MuJoCo in the wheel
would have made every PartCAD release a statement about which MuJoCo you get,
and MJCF is MuJoCo's.

`urdf` is the counter-example and stays in PartCAD's `//builtin/import`: a URDF
describes a robot rather than any one engine's world, and ROS, MuJoCo, PyBullet
and Isaac all read it.

Once this package is listed in the [public
index](https://github.com/partcad/partcad-index) it will also be reachable as
`//pub/feature/simulate/mujoco` — which is the name it gives itself — without
the `dependencies:` entry above.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
