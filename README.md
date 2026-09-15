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
| `open:` | open one in the MuJoCo viewer | `pc open --with mujoco` |

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

PartCAD exports the scene as MJCF with every
body free to move and a ground plane under it, hands this package the file, and
this package steps the model for `duration` seconds of simulated time. It then
reports where every body was at the start and at the end:

```json
{
  "before": {"time": 0.0,  "bodies": {"top": {"pos": [0, 0, 30], "quat": [1, 0, 0, 0]}}},
  "after":  {"time": 10.0, "bodies": {"top": {"pos": [29.7, 0, 9.8], "quat": [...]}}},
  "simulator": "mujoco", "duration": 10.0, "steps": 5000, "units": "mm"
}
```

Positions are in **millimetres**, PartCAD's unit everywhere, not the metres
MuJoCo works in: a `validation:` expression is written by whoever wrote the
part, against the numbers that part is drawn in.

PartCAD reads nothing inside `before` and `after`. It hands them to the
`validation:` expression the package wrote and reports what that says — every
judgement in that sentence belongs to the package being simulated.

Nothing has to be installed by hand. PartCAD installs this package's Python
requirements into the sandbox it runs the implementation in, so a machine with
no MuJoCo on it simulates just the same.

## Parameters

Set any of these as fields of the simulation (per package), or in the `params:`
of one `simulate:` entry (per simulation).

| Parameter | Default | What it is |
| --- | --- | --- |
| `duration` | `10.0` | Seconds of simulated time to run for. |
| `timestep` | MuJoCo's | Integration step, in seconds. |
| `gravity` | `[0, 0, -9.81]` | m/s², in the scene's frame. |
| `samples` | `0` | Report the state at this many evenly spaced instants too, as `samples`. |

## Friction is a fact about the material

Whether a stack of blocks stands up is not a property of its geometry. Two 20 mm
cubes squarely stacked, ten seconds under gravity, nothing changed but the
coefficient the blocks are given:

| sliding friction | what happens to the top block |
| --- | --- |
| 0.04 (PTFE) | slides off and ends up on the floor — 35 mm |
| 0.4 | the same |
| 0.5 | it stays — 3 mm of settling |
| 1.05 (dry aluminium) | it stays — 1 mm |

So state it. A part that declares a material whose `mu` is set gets that
coefficient written into the MJCF, and the simulation answers for the material
the part is actually made of. A part that states neither gets MuJoCo's default
of 1.0 — a plausible number for metal on metal, a badly wrong one for PTFE, and
in either case a number nobody chose.

PartCAD writes each body's own coefficient and says nothing about how the two
sides of a contact combine: that is MuJoCo's model rather than the part's.

## Tests

```shell
pytest
```

`test_mjcf.py` needs `partcad` installed: the reader and the writer are written
against the sandbox contract that lives there — `ocp_serialize`, `urdf_common`
and `primitive_shapes` — and the tests import them the same way a sandbox does.
It needs no MuJoCo, because reading and writing a model do not involve one.

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
