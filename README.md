# partcad-sim-mujoco

A [PartCAD](https://partcad.org/) package that **runs a PartCAD scene in
[MuJoCo](https://mujoco.org/)**, so that a part or an assembly can state what it
is supposed to do once the world is switched on — and have that checked.

```yaml
dependencies:
  sim-mujoco:
    type: git
    url: https://github.com/partcad/partcad-sim-mujoco.git

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

## What it does

PartCAD exports the scene as **MJCF** — MuJoCo's own model format — with every
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
blocks squarely stacked, ten seconds under gravity, nothing else changed:

| sliding friction | what happens |
| --- | --- |
| 0.0 | the stack scatters — 172 mm |
| 0.05 | the top block slides off — 62 mm |
| 0.2 | nothing moves — 0.3 mm |
| 1.0 | nothing moves — 1.0 mm |

So state it. A part that declares a `material:` whose material states `mu` gets
that coefficient written into the MJCF, and the simulation answers for the
material the part is actually made of. A part that states neither gets MuJoCo's
default of 1.0 — a plausible number for metal on metal, a badly wrong one for
PTFE, and in either case a number nobody chose.

## Where this came from

This used to ship inside the `partcad` wheel as `//builtin/simulate`. It does
not any more, and should not: PartCAD ships the *concept* — the `simulation:`
section, the sandbox wrapper, the runner, the MJCF export — and a simulator is
somebody's program with a release cycle of its own. Pinning MuJoCo in the wheel
would have made every PartCAD release a statement about which MuJoCo you get.

Once this package is listed in the [public
index](https://github.com/partcad/partcad-index) it will also be reachable as
`//pub/feature/simulate/mujoco` — which is the name it gives itself — without
the `dependencies:` entry above.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
