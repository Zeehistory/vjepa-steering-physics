# Steering Physics in World Models

Code accompanying the ICLR submission *Steering Physics in World Models*.
This repository contains the full pipeline used in the paper, from generating the videos to the
results:

1. **Controlled physics videos.** Procedurally generated 2-D scenes and MuJoCo 3-D scenes in which a
   single physical quantity (velocity, acceleration, angular velocity, restitution) varies while
   position, appearance and everything else stay fixed.
2. **Frozen encoders.** V-JEPA 2 (ViT-L / ViT-H / ViT-g) latents cached layer by layer.
3. **Reading.** Probes that measure how decodable each quantity is, with shuffled-latent and
   randomized-label controls.
4. **Writing.** Linear command operators, Fourier / polar operators for rotational quantities,
   spline and transport operators for second-order quantities, and test-time optimisation (TTO)
   baselines that edit a latent so that it encodes a target value.
5. **Decoding.** Latent-to-pixel decoders, used to check that an edit changes the rendered video
   and not only the latent.
6. **Acting.** A MuJoCo paddle / Franka Panda strike task in which a steered latent is converted to an
   action. The action is executed in simulation and the outcome is measured in m/s.

The encoder is frozen throughout. The only models trained are the probes, the operators, the pixel
decoders and the inverse action predictor.

---

## Installation

```bash
conda env create -f environment.yml         # Python 3.11, installs this package in editable mode
conda activate vjepa-physics-decoder
# CPU-only smoke test (mock encoder, no weight download):
pip install -e ".[dev]"
```

Optional extras (declared in `pyproject.toml`): `encoders` (V-JEPA 2 weights via Hugging Face),
`extras` (MuJoCo, rendering, plotting). The MuJoCo scenes render headless with `MUJOCO_GL=egl`.
The robotics scenes additionally need the Franka Panda model from MuJoCo Menagerie:
`bash experiments/threads/paddle-robotics/01_data/fetch_menagerie.sh`.

## Quick check

```bash
PYTHONPATH=. pytest tests -q                 # unit + integration tests
python experiments/pipeline/02_encode/extract_latents.py \
    --config configs/train/smoke_synthetic.yaml --output_dir outputs/smoke/latents   # offline mock run
```

---

## Repository layout

```
src/                     library code (importable as `src`)
  data/                  scene generators: moving_ball (2-D scenes), rolling_ball3d, spin_ball3d,
                         paddle_strike, franka_strike, franka_dynamic_strike, physics_iq, ...
  encoders/              frozen V-JEPA / V-JEPA 2 wrappers, layer hooks, latent cache I/O
  training/              probes (linear ridge + MLP, with controls), decoder training loop
  analysis/              steering operators (velocity_ops, spline_ops, manifold_ops, spin_ops,
                         fourier_orientation, subspace, intervention), latent geometry
  decoders/              latent-to-pixel decoders and losses
  control/               strike_inverse (analytic inverse), action_predictor (learned inverse policy)
  eval/, utils/          metrics, config and I/O helpers
configs/                 data / encoder / decoder / train / analysis YAMLs
experiments/
  pipeline/<NN_stage>/   reusable stages shared by all experiments
  threads/<topic>/<NN_stage>/   experiment-specific scripts, one folder per paper section
tests/                   pytest suite
docs/                    per-experiment protocols
```

Stages are numbered the same way everywhere:

| stage | contents |
|---|---|
| `00_common` | shared shell helpers |
| `01_data` | dataset generation and validation (label and tracker certification) |
| `02_encode` | latent extraction with the frozen encoder |
| `03_train` | probe and decoder training |
| `04_operators` | fitting steering operators |
| `05_steering` | applying edits and decoding the result |
| `06_probes` | readability probes and controls |
| `07_eval` | evaluation and ablations |
| `08_figures` | figures, filmstrips and videos |
| `10_collect`, `11_run` | result aggregation and end-to-end drivers |

Python entry points take `--help`. The `.sh` files are SLURM job scripts. They use generic `#SBATCH`
resources and no partition, so pass `--partition` / `--account` for your cluster on the command line
(`sbatch -p <partition> script.sh`). All scripts are run from the repository root with
`PYTHONPATH=.`, and write to `outputs/` by default.

---

## Experiments ↔ paper

| paper topic | folder | main entry points |
|---|---|---|
| Datasets and encoding | `experiments/pipeline/01_data`, `02_encode` | `generate_synthetic.py`, `validate_scene_data.py`, `extract_latents.py` |
| Reading physics (probes) | `experiments/pipeline/06_probes`, `threads/velocity/06_probes`, `threads/acceleration/06_probes` | `probe_velocity.py`, `probe_accel.py`, `probe_occlusion.py`, `probe_equivariance.py` |
| Velocity steering | `experiments/threads/velocity` | `04_operators/speed_axis.py`, `05_steering/steer_velocity2d.py`, `11_run/step2_velocity_pipeline.sh` |
| Command operators (shared) | `experiments/pipeline/04_operators` | `fit_command_operators.py`, `calibrate_cmd_gain.py`, `probe_standardized_ridge.py` |
| Acceleration steering | `experiments/threads/acceleration` | `05_steering/steer_accel2d.py`, `steer_accel_spline.py`, `steer_accel_decopt.py` |
| Angular velocity / acceleration | `experiments/threads/angular-velocity` | `05_steering/steer_angvel_fourier.py`, `steer_angvel_polar.py`, `06_probes/probe_angvel_read.py` |
| Restitution and spin | `experiments/threads/restitution-spin` | `04_operators/fit_spin_operators.py`, `fit_transport_spin.py`, `06_probes/spin_probe.py` |
| Encoder scale (ViT-L/H/g) | `experiments/threads/model-size-sweep` | `11_run/decoded_sweep.sh`, `06_probes/bootstrap_gate.py` |
| Latent-to-pixel decoder | `experiments/threads/hifi-decoder`, `pipeline/03_train` | `03_train/train_hifi.sh`, `pipeline/03_train/train_decoder.py`, `pipeline/07_eval/eval_decoder_fidelity.py` |
| From steering to action (robotics) | `experiments/threads/paddle-robotics` | see [`docs/robotics_protocol.md`](docs/robotics_protocol.md) |

### Typical workflow (velocity example)

```bash
# 1. generate + encode (GPU): videos are generated on the fly from fixed seeds
DATASET=moving_ball_velocity sbatch experiments/pipeline/02_encode/extract_ball.sh
for s in train val test; do SPLIT=$s sbatch experiments/pipeline/02_encode/extract_scene.sh; done
# 2. read: layer-wise probe (linear + MLP) with controls
sbatch experiments/threads/velocity/06_probes/probe_velocity.sh
# 3. write: fit command operators, then steer, decode and track in pixels
sbatch experiments/pipeline/04_operators/fit_command.sh
sbatch experiments/threads/velocity/05_steering/steer_v2d.sh
```

Each job script documents its environment variables (dataset, layers, output directory) in its
header.

### Robotics (steering → action)

The full procedure (scene, dataset, encoding, steering map, inverse policy, closed-loop execution,
baselines and controls) is described in [`docs/robotics_protocol.md`](docs/robotics_protocol.md).
Short version:

```bash
python experiments/pipeline/01_data/validate_paddle_strike.py --output_dir outputs/cert_paddle   # CPU, ~15 s
EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/02_encode/extract_paddle_strike.sh
EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/03_train/train_action_predictor.sh
```

---

## Evaluation conventions

- **Held-out scenes.** Every split is scene-disjoint (distinct generator seeds). Where clips share an
  episode, bootstrap intervals resample scenes, not clips.
- **Controls are always on.** Probes report shuffled-latent and randomized-label controls. Steering
  reports no-op, wrong-target and shuffled baselines next to the method.
- **Labels are measured, not commanded.** Physical labels come from the simulator state or a pixel
  tracker, never from the value that was requested.
- **Pixels, not only latents.** Steering claims are checked on decoded frames with an
  independent tracker. Latent-space agreement alone is reported only as a diagnostic.

## Compute

The experiments ran on SLURM clusters with single NVIDIA GPUs (RTX 6000 Ada / H200 / B200 class).
Latent extraction and decoder training need a GPU. Probes, operator fitting, the robotics certificate
and the action predictor run on CPU. Cached latents are large (tens to hundreds of GB per dataset
at full resolution), so point `outputs/` at a scratch filesystem.

## License

Apache-2.0 (see `LICENSE`). Model weights are downloaded from their original sources and are
subject to their own licenses.
